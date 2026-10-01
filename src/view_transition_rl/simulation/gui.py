"""Desktop VTK UI. Imported only by the gui subcommand."""

from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import vtk

from .environment import PHANTOMS, PhantomEnv
from .geometry import euler_from_pose, pose_from_euler


class PhantomViewer:
    def __init__(self, env, *, off_screen=False, export_dir="outputs/simulation/captures"):
        try:
            import pyvista as pv
        except ImportError as exc:
            raise RuntimeError('Install the GUI extra: pip install ".[gui]"') from exc
        self.pv, self.env = pv, env
        self.export_dir = Path(export_dir)
        self.pending = None
        self.last_slice_ms = 0.0
        self.notice = "Ready. Bottom centre stays on the surface."
        self.building = True
        self.widgets = {}
        self.plotter = pv.Plotter(shape=(1, 3), window_size=(1560, 960), border=False,
                                  off_screen=off_screen, title="Vessel Phantom | 5-DOF Probe")
        p = self.plotter
        for index, viewport in enumerate([(0, 0, .58, 1), (.58, .36, 1, 1), (.58, 0, 1, .36)]):
            p.renderers[index].SetViewport(*viewport)
        p.subplot(0, 0)
        p.set_background("#111d2b", all_renderers=False)
        width, length, depth = env.config.phantom_size_mm
        probe_path = env.assets / "probe_model_50mm.stl"
        self.probe_mesh = pv.read(probe_path)
        self.probe_actor = p.add_mesh(self.probe_mesh, color="#d2dbe5", smooth_shading=True,
                                      ambient=.22, diffuse=.78, specular=.15,
                                      name="probe", reset_camera=False)
        p.add_text("VESSEL PHANTOM", position=(.035, .94), viewport=True,
                   font_size=23, color="#f0f5fc", name="title")
        p.add_text("Drag: orbit   Wheel: zoom\n"
                   "Arrows: local x/y 0.5 mm   A/D: local Rx 1 deg\n"
                   "W/S: local Ry 1 deg   Q/E: local Rz 1 deg\n"
                   "R: reset pose to zero   P: export",
                   position=(.035, .025), viewport=True, font_size=10, color="#b8c9da", name="help")
        p.camera_position = [(125, -175, -135), (0, 0, 10), (0, 0, -1)]
        p.camera.parallel_projection = True
        p.camera.parallel_scale = 86

        p.subplot(0, 1)
        p.set_background("#0a111c", all_renderers=False)
        h, w = env.config.image_shape
        fw, fd = env.config.fov_mm
        self.mask_grid = pv.ImageData(dimensions=(w, h, 1), spacing=(fw / w, fd / h, 1),
                                      origin=(-fw / 2 + fw / w / 2, fd / h / 2, 0))
        self.mask_grid.point_data["mask"] = np.zeros(h * w, dtype=np.uint8)
        p.add_mesh(self.mask_grid, scalars="mask", cmap="gray", clim=(0, 255),
                   show_scalar_bar=False, lighting=False, name="mask", reset_camera=False)
        p.add_text("LIVE VESSEL MASK", position=(.04, .92), viewport=True,
                   font_size=18, color="#f0f5fc", name="mask_title")
        p.add_text(f"{w} x {h} px  |  {fw:g} x {fd:g} mm  |  probe y = 0",
                   position=(.04, .865), viewport=True, font_size=10, color="#9eb5ca")
        p.add_text(f"Top: depth 0 mm     Columns: probe +x >\nBottom: depth +{fd:g} mm (probe +z)",
                   position=(.04, .025), viewport=True, font_size=10, color="#9eb5ca", name="mask_axes")
        p.view_xy()
        p.camera.parallel_projection = True
        p.camera.focal_point = (0, fd / 2, 0)
        p.camera.position = (0, fd / 2, 100)
        p.camera.parallel_scale = max(fd / .72, fw / .8) / 2

        p.subplot(0, 2)
        p.set_background("#182637", all_renderers=False)
        p.add_text("POSE CONTROLS", position=(.04, .90), viewport=True,
                   font_size=15, color="#f0f5fc")
        state = env.get_state()
        pose = np.array(state["phantom_T_probe_mm"])
        angles = np.rad2deg(euler_from_pose(pose))
        values = [pose[0, 3], pose[1, 3], *angles]
        specs = [("x", "Surface X / mm", (-width / 2, width / 2), (.06, .44, .71)),
                 ("y", "Surface Y / mm", (-length / 2, length / 2), (.57, .95, .71)),
                 ("roll", "Roll / deg", (-60, 60), (.06, .29, .46)),
                 ("pitch", "Pitch / deg", (-60, 60), (.39, .62, .46)),
                 ("yaw", "Yaw / deg", (-180, 180), (.72, .95, .46))]
        for (key, label, limits, (start, end, ypos)), value in zip(specs, values):
            widget = p.add_slider_widget(lambda val, k=key: self._slider_changed(k, val), limits,
                                          value=value, title=label, color="#72cbd3", fmt="%.1f",
                                          interaction_event="always", title_height=.022,
                                          slider_width=.016, tube_width=.004)
            self._position_slider(widget, start, end, ypos)
            self.widgets[key] = widget
        self._add_reset_button()
        p.subplot(0, 0)
        self.phantom_button = self._make_button(
            tuple(f"PHANTOM {i} / SWITCH" for i in range(1, len(PHANTOMS) + 1)),
            self.toggle_phantom)
        self._sync_phantom_button()
        self._position_phantom_button()
        p.iren.add_observer("ConfigureEvent", self._position_phantom_button)
        self._update_phantom_scene()
        self.building = False
        self._bind_keys()
        self.refresh()
        if not off_screen:
            # Only the newest slider request is sampled; no growing work queue.
            p.iren.add_observer("TimerEvent", self._timer)
            self.timer_id = p.iren.create_timer(33)

    def toggle_phantom(self):
        phantoms = list(PHANTOMS)
        current = phantoms.index(self.env.phantom_id)
        return self.switch_phantom(phantoms[(current + 1) % len(phantoms)])

    def _sync_phantom_button(self):
        self.phantom_button.GetRepresentation().SetState(list(PHANTOMS).index(self.env.phantom_id))

    def _position_phantom_button(self, *_):
        width, height = self.plotter.window_size
        viewport = self.plotter.renderers[0].GetViewport()
        panel_width = (viewport[2] - viewport[0]) * width
        panel_height = (viewport[3] - viewport[1]) * height
        self.phantom_button.GetRepresentation().PlaceWidget([
            .035 * panel_width, .36 * panel_width,
            .735 * panel_height, .79 * panel_height, 0, 0])

    def switch_phantom(self, phantom):
        """Validate the replacement before changing the active environment."""
        if phantom == self.env.phantom_id:
            self._sync_phantom_button()
            return True
        try:
            replacement = PhantomEnv(phantom=phantom, config_overrides=self.env.config_overrides)
            replacement.get_mask()
        except (ValueError, TypeError, OSError) as exc:
            self.notice = f"Cannot switch phantom: {exc}"
            self._sync_phantom_button()
            self.refresh()
            return False
        self.pending = None
        self.env = replacement
        self._sync_phantom_button()
        self._update_phantom_scene()
        self.notice = f"Selected {replacement.definition['name']}"
        self.refresh()
        return True

    def _update_phantom_scene(self):
        p, pv, env = self.plotter, self.pv, self.env
        p.subplot(0, 0)
        width, length, depth = env.config.phantom_size_mm
        box = pv.Box(bounds=(-width / 2, width / 2, -length / 2, length / 2, 0, depth))
        p.add_mesh(box, color="#77b9c6", opacity=.10, name="phantom", reset_camera=False, render=False)
        p.add_mesh(box, style="wireframe", color="#6c99a8", line_width=1.5,
                   name="outline", reset_camera=False, render=False)
        top = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=width,
                       j_size=length, i_resolution=7, j_resolution=10)
        p.add_mesh(top, style="wireframe", color="#365362", opacity=.45,
                   name="surface", reset_camera=False, render=False)
        p.add_mesh(pv.wrap(env.mesh), color="#ef6c78", smooth_shading=True,
                   name="vessel", reset_camera=False, render=False)
        p.add_text(f"{env.definition['name']}  {width:g} x {length:g} x {depth:g} mm\n"
                   "Centre contact | 5 DOF | geometric segmentation",
                   position=(.035, .875), viewport=True, font_size=11,
                   color="#9eb5ca", name="subtitle", render=False)
        for key, size in [("x", width), ("y", length)]:
            rep = self.widgets[key].GetRepresentation()
            rep.SetMinimumValue(-size / 2)
            rep.SetMaximumValue(size / 2)
        p.camera_position = [(125, -175, -135), (0, 0, 10), (0, 0, -1)]
        p.camera.parallel_projection = True
        p.camera.parallel_scale = max(86, length * .8, width * .8, depth * 1.5)

    def _add_reset_button(self):
        self.reset_button = self._make_button("RESET TO ZERO", self.reset)
        self._position_reset_button()
        self.plotter.iren.add_observer("ConfigureEvent", self._position_reset_button)

    def _make_button(self, label, callback):
        import matplotlib
        import vtk
        from PIL import Image, ImageDraw, ImageFont
        from vtk.util.numpy_support import numpy_to_vtk

        font = ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"), 26)
        representation = vtk.vtkTexturedButtonRepresentation2D()
        labels = (label, label) if isinstance(label, str) else tuple(label)
        representation.SetNumberOfStates(len(labels))
        for index, text in enumerate(labels):
            background = "#285969" if index == 0 else "#387a88"
            image = Image.new("RGB", (360, 70), background)
            draw = ImageDraw.Draw(image)
            draw.rounded_rectangle((1, 1, 358, 68), radius=9, outline="#72cbd3", width=2)
            draw.text((180, 35), text, font=font, anchor="mm", fill="#f0f5fc")
            texture = vtk.vtkImageData()
            texture.SetDimensions(360, 70, 1)
            texture.GetPointData().SetScalars(numpy_to_vtk(np.asarray(image)[::-1].reshape(-1, 3), deep=True))
            representation.SetButtonTexture(index, texture)
        representation.SetPlaceFactor(1)
        button = vtk.vtkButtonWidget()
        button.SetInteractor(self.plotter.iren.interactor)
        button.SetCurrentRenderer(self.plotter.renderer)
        button.SetRepresentation(representation)
        button.AddObserver("StateChangedEvent", lambda *_: callback())
        button.On()
        return button

    def _position_reset_button(self, *_):
        width, height = self.plotter.window_size
        viewport = self.plotter.renderers[2].GetViewport()
        panel_width = (viewport[2] - viewport[0]) * width
        panel_height = (viewport[3] - viewport[1]) * height
        # Button placement is in renderer-local pixels; event hit-testing uses
        # full-window coordinates. Do not add the viewport origin here.
        self.reset_button.GetRepresentation().PlaceWidget([
            .26 * panel_width, .74 * panel_width,
            .10 * panel_height, .28 * panel_height, 0, 0])

    @staticmethod
    def _position_slider(widget, start, end, y):
        rep = widget.GetRepresentation()
        viewport = widget.GetCurrentRenderer().GetViewport()
        # VTK slider geometry uses renderer-local display coordinates. Adding
        # viewport origin here places widgets outside their renderer in VTK 9.6.
        for coordinate, x in [(rep.GetPoint1Coordinate(), start), (rep.GetPoint2Coordinate(), end)]:
            coordinate.SetCoordinateSystemToNormalizedDisplay()
            coordinate.SetValue(x * (viewport[2] - viewport[0]), y * (viewport[3] - viewport[1]))
        rep.SetTitleHeight(.060)
        rep.SetLabelHeight(.060)

    def _slider_changed(self, key, value):
        if self.building:
            return
        # Read all widgets so concurrent drags keep the most recent requested pose.
        values = {k: w.GetRepresentation().GetValue() for k, w in self.widgets.items()}
        values[key] = value
        self.pending = pose_from_euler(values["x"], values["y"],
                                       *np.deg2rad([values["roll"], values["pitch"], values["yaw"]]))

    def _bind_keys(self):
        p = self.plotter
        # Real keystrokes produce KeyPressEvent followed by CharEvent. Clearing
        # PyVista's key callbacks alone leaves VTK's OnChar defaults active:
        # Q/E exit, W/S change actor representation, R resets the camera.
        # An observer ON THE STYLE replaces its native OnChar handling while
        # preserving mouse interaction and the application's key callbacks.
        p.iren.clear_key_event_callbacks()
        p.iren.style.add_observer("CharEvent", self._ignore_vtk_character_shortcuts)
        actions = {
            "Left": [-.5, 0, 0, 0, 0], "Right": [.5, 0, 0, 0, 0],
            "Down": [0, -.5, 0, 0, 0], "Up": [0, .5, 0, 0, 0],
        }
        for keys, index in [(('a', 'd'), 2), (('w', 's'), 3), (('q', 'e'), 4)]:
            for key, sign in zip(keys, [-1, 1]):
                action = [0.0] * 5
                action[index] = sign * np.deg2rad(1)
                actions[key] = action
        for key, action in actions.items():
            p.add_key_event(key, lambda a=action: self.move(a))
            if len(key) == 1:
                p.add_key_event(key.upper(), lambda a=action: self.move(a))
        for key, callback in [("r", self.reset), ("p", self.export)]:
            p.add_key_event(key, callback)
            p.add_key_event(key.upper(), callback)

    @staticmethod
    def _ignore_vtk_character_shortcuts(*_):
        """Only the application bindings act on character keys."""
        pass

    def _timer(self, *_):
        self.flush_pending()

    def flush_pending(self):
        if self.pending is None:
            return
        pose = self.pending
        self.pending = None
        self._apply_result(self.env.set_pose(pose), "Pose updated")

    def _apply_result(self, result, message):
        self.notice = message if result.accepted else "Rejected: " + result.reason
        self.refresh()

    def move(self, action):
        self.flush_pending()
        self._apply_result(self.env.step(action), "Local motion applied")

    def reset(self):
        self.pending = None
        self.reset_button.GetRepresentation().SetState(0)
        self._apply_result(self.env.set_pose(np.eye(4)), "Pose reset: X / Y / Roll / Pitch / Yaw = 0")

    def export(self):
        self.flush_pending()
        prefix = self.export_dir / datetime.now().strftime("capture_%Y%m%d_%H%M%S_%f")
        try:
            paths = self.env.export(prefix)
            self.notice = "Saved " + paths[0].name + " and pose JSON"
            print("Exported:", *paths)
        except OSError as exc:
            self.notice = f"Export failed: {exc}"
        self.refresh()

    def refresh(self):
        p, pv = self.plotter, self.pv
        state = self.env.get_state()
        pose = np.array(state["phantom_T_probe_mm"])
        p.subplot(0, 0)
        lx, ly = np.array(self.env.config.probe_size_mm) / 2
        probe_matrix_values = pose.copy()
        probe_matrix_values[:3, 0] *= self.env.config.probe_size_mm[0] / 40.0
        probe_matrix_values[:3, 1] *= self.env.config.probe_size_mm[1] / 10.0
        probe_matrix = vtk.vtkMatrix4x4()
        for row in range(4):
            for column in range(4):
                probe_matrix.SetElement(row, column, float(probe_matrix_values[row, column]))
        self.probe_actor.SetUserMatrix(probe_matrix)
        base = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=2 * lx,
                        j_size=2 * ly, i_resolution=1, j_resolution=1).transform(pose, inplace=False)
        p.add_mesh(base, color="#47c5d0", opacity=.7, name="probe_bottom", reset_camera=False, render=False)
        fw, fd = self.env.config.fov_mm
        points = np.array([[-fw / 2, 0, 0], [fw / 2, 0, 0], [fw / 2, 0, fd], [-fw / 2, 0, fd]])
        plane = pv.PolyData(points @ pose[:3, :3].T + pose[:3, 3], [4, 0, 1, 2, 3])
        p.add_mesh(plane, color="#e4b775", opacity=.20, show_edges=True,
                   name="image_plane", reset_camera=False, render=False)
        for i, (label, color) in enumerate(zip("xyz", ["#fc777a", "#8adb98", "#79b4ff"])):
            arrow = pv.Arrow(start=pose[:3, 3], direction=pose[:3, i], scale=15)
            p.add_mesh(arrow, color=color, name="axis_" + label, reset_camera=False, render=False)
        p.add_text("Probe axes:  x red / long edge   y green / short edge   z blue / inward",
                   position=(.035, .82), viewport=True, font_size=10, color="#b8c9da", name="axes_legend", render=False)
        start = perf_counter()
        mask = self.env.get_mask()
        self.last_slice_ms = (perf_counter() - start) * 1000
        self.mask_grid.point_data["mask"] = mask[::-1].ravel()
        p.add_text(f"X {pose[0,3]:+.2f} mm   Y {pose[1,3]:+.2f} mm   Tilt {state['tilt_deg']:.1f} deg\n"
                   f"{self.notice}", position=(.035, .14), viewport=True, font_size=11,
                   color="#efb2a9" if not state["last_action_accepted"] else "#72cbd3", name="status", render=False)
        angles = np.rad2deg(euler_from_pose(pose))
        for key, value in zip(["x", "y", "roll", "pitch", "yaw"],
                              [pose[0, 3], pose[1, 3], *angles]):
            self.widgets[key].GetRepresentation().SetValue(float(value))
        p.subplot(0, 1)
        p.add_text(f"Foreground {(mask > 0).sum():,} px   |   Mask update {self.last_slice_ms:.1f} ms",
                   position=(.04, .805), viewport=True, font_size=10, color="#72cbd3", name="mask_stats", render=False)
        p.subplot(0, 0)
        p.render()

    def close(self):
        self.plotter.close()


def run_gui(env, screenshot=None, export_dir="outputs/simulation/captures"):
    viewer = PhantomViewer(env, off_screen=bool(screenshot), export_dir=export_dir)
    if screenshot:
        path = Path(screenshot)
        path.parent.mkdir(parents=True, exist_ok=True)
        viewer.plotter.show(screenshot=str(path), auto_close=True)
        print(path)
    else:
        viewer.plotter.show()
