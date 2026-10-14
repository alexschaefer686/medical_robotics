"""3D view of phantom box, vessel and probe beside the live mask.

Unlike gui.py there are no sliders: a controller drives the probe and update()
redraws; callers may bind keys with on_key(). Requires the "gui" extra (PyVista).

body_offset shifts the drawn box and vessel (and the probe with them) so that
body motion, simulated as an opposite probe shift, appears as the body moving.
"""

from pathlib import Path

import numpy as np


def _import_pyvista():
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError('Install the GUI extra: pip install ".[gui]"') from exc
    return pv


class _Window:
    """Shared window handling; subclasses create self.plotter."""

    _shown = False

    def on_key(self, key, callback):
        """Bind a VTK key symbol (e.g. "Left", "Up", "w") to callback(), replacing PyVista's default
        binding (Up/Down zoom the camera by default). Letters are bound in both cases, and VTK's
        own letter shortcut for them (e.g. W wireframe, S surface, E quit) is suppressed."""
        keys = [key, key.upper()] if len(key) == 1 and key.isalpha() else [key]
        for k in keys:
            self.plotter.iren.clear_events_for_key(k)
            self.plotter.add_key_event(k, callback)
        if len(key) == 1:
            if not hasattr(self, "_bound_chars"):
                self._bound_chars = set()
                # An observer on the style replaces its OnChar handling; unbound keys are passed on.
                self.plotter.iren.style.AddObserver("CharEvent", self._on_char)
            self._bound_chars.update(keys)

    def _on_char(self, style, _event):
        if self.plotter.iren.interactor.GetKeySym() not in self._bound_chars:
            style.OnChar()

    def show(self, block=False):
        """Open the window. block=False returns immediately so a loop can keep updating."""
        if not block and not self._shown:
            self.plotter.show(interactive_update=True, auto_close=False)
            self._shown = True
        elif block:
            self.plotter.show()

    @property
    def is_open(self):
        p = self.plotter
        if p._closed or p.render_window is None:
            return False
        # The window's close button ends the interactor without closing the plotter.
        return p.off_screen or not self._shown or not p.iren.interactor.GetDone()

    def process_events(self):
        if self.is_open and not self.plotter.off_screen:  # off-screen windows have no event loop
            self.plotter.update()

    def screenshot(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.plotter.screenshot(str(path))
        return path

    def close(self):
        self.plotter.close()


class SceneView(_Window):
    def __init__(self, env, *, off_screen=False, window_size=(1400, 700)):
        pv = _import_pyvista()
        self.pv, self.env = pv, env
        self.plotter = pv.Plotter(shape=(1, 2), window_size=window_size, off_screen=off_screen,
                                  title="Vessel Phantom | probe simulation")
        p = self.plotter
        width, length, depth = env.config.phantom_size_mm

        p.subplot(0, 0)
        p.set_background("#111d2b", all_renderers=False)
        box = pv.Box(bounds=(-width / 2, width / 2, -length / 2, length / 2, 0, depth))
        self.body_offset = np.zeros(2)  # displayed phantom x/y displacement, mm
        self.body_actors = [
            p.add_mesh(box, color="#77b9c6", opacity=.10, reset_camera=False),
            p.add_mesh(box, style="wireframe", color="#6c99a8", line_width=1.5, reset_camera=False),
            p.add_mesh(pv.wrap(env.mesh), color="#ef6c78", smooth_shading=True, reset_camera=False),
        ]
        self._text = ""
        self.probe_mesh = pv.read(env.assets / "probe_model_50mm.stl")
        self.probe_actor = p.add_mesh(self.probe_mesh, color="#d2dbe5", smooth_shading=True,
                                      ambient=.22, diffuse=.78, specular=.15, reset_camera=False)
        p.add_text("Probe axes:  x red / long edge   y green / short edge   z blue / inward",
                   position=(.02, .02), viewport=True, font_size=10, color="#b8c9da")
        # +z points into the phantom, so the camera looks from below with -z up.
        p.camera_position = [(125, -175, -135), (0, 0, 10), (0, 0, -1)]
        p.camera.parallel_projection = True
        p.camera.parallel_scale = max(86, length * .8, width * .8, depth * 1.5)

        p.subplot(0, 1)
        p.set_background("#0a111c", all_renderers=False)
        h, w = env.config.image_shape
        fw, fd = env.config.fov_mm
        self.mask_grid = pv.ImageData(dimensions=(w, h, 1), spacing=(fw / w, fd / h, 1),
                                      origin=(-fw / 2 + fw / w / 2, fd / h / 2, 0))
        self.mask_grid.point_data["mask"] = np.zeros(h * w, dtype=np.uint8)
        p.add_mesh(self.mask_grid, scalars="mask", cmap="gray", clim=(0, 255),
                   show_scalar_bar=False, lighting=False, reset_camera=False)
        p.view_xy()
        p.camera.parallel_projection = True
        p.camera.focal_point = (0, fd / 2, 0)
        p.camera.position = (0, fd / 2, 100)
        p.camera.parallel_scale = max(fd, fw) / 2 * 1.15
        self.update()

    def update(self, text=None, boundary=None):
        """Redraw body, probe, image plane and mask for the current simulator pose.

        text: status text; None keeps the previous one.
        boundary: optional bool node array (H-1, W-1) from
        visual_servoing.features.boundary_nodes, drawn in red over the mask.
        """
        p, pv, env = self.plotter, self.pv, self.env
        if text is not None:
            self._text = text
        shift = np.eye(4)
        shift[:2, 3] = self.body_offset
        for actor in self.body_actors:
            actor.user_matrix = shift
        pose = shift @ np.array(env.get_state()["phantom_T_probe_mm"])
        p.subplot(0, 0)
        # The probe STL is modelled for a 40x10 mm footprint.
        scaled = pose.copy()
        scaled[:3, 0] *= env.config.probe_size_mm[0] / 40.0
        scaled[:3, 1] *= env.config.probe_size_mm[1] / 10.0
        self.probe_actor.user_matrix = scaled
        fw, fd = env.config.fov_mm
        corners = np.array([[-fw / 2, 0, 0], [fw / 2, 0, 0], [fw / 2, 0, fd], [-fw / 2, 0, fd]])
        plane = pv.PolyData(corners @ pose[:3, :3].T + pose[:3, 3], [4, 0, 1, 2, 3])
        p.add_mesh(plane, color="#e4b775", opacity=.25, show_edges=True,
                   name="image_plane", reset_camera=False, render=False)
        for i, (label, color) in enumerate(zip("xyz", ["#fc777a", "#8adb98", "#79b4ff"])):
            # Longer than half the probe so x and y protrude from the probe body.
            arrow = pv.Arrow(start=pose[:3, 3], direction=pose[:3, i], scale=30)
            p.add_mesh(arrow, color=color, name="axis_" + label, reset_camera=False, render=False)
        p.add_text(self._text, position="upper_left", font_size=10, color="#d6e2ee", name="status", render=False)
        p.subplot(0, 1)
        self.mask_grid.point_data["mask"] = env.get_mask()[::-1].ravel()
        self._draw_boundary(boundary)
        p.render()

    def _draw_boundary(self, boundary):
        p = self.plotter
        if boundary is None or not np.any(boundary):
            p.remove_actor("boundary", render=False)
            return
        h, w = self.env.config.image_shape
        fw, fd = self.env.config.fov_mm
        rows, cols = np.nonzero(boundary)
        # Node (i, j) lies at pixel (row i + 0.5, col j + 0.5); mask row 0 is drawn at the top.
        points = np.column_stack([-fw / 2 + (cols + 1) * fw / w, fd - (rows + 1) * fd / h,
                                  np.full(rows.size, 0.1)])
        p.add_mesh(self.pv.PolyData(points), color="red", point_size=3, lighting=False,
                   name="boundary", reset_camera=False, render=False)

    def set_help(self, text):
        self.plotter.subplot(0, 0)
        self.plotter.add_text(text, position=(.02, .055), viewport=True, font_size=10,
                              color="#b8c9da", name="help", render=False)



class PointCloudView(_Window):
    """Separate window accumulating 3D points (e.g. vessel boundary) over the steps.

    add() collects points; draw_points() displays all of them or a chosen subset,
    e.g. only the points a fit used. With a simulator env, the vessel mesh and
    phantom outline are drawn faintly for reference; body_offset shifts them like
    in SceneView. env=None (e.g. on a real robot) shows only the data, and the
    camera frames the first points drawn. Points are coloured by when they were recorded.
    """

    def __init__(self, env=None, *, off_screen=False, window_size=(800, 700), title="Boundary point cloud"):
        pv = _import_pyvista()
        self.pv, self.env = pv, env
        self.plotter = p = pv.Plotter(window_size=window_size, off_screen=off_screen, title=title)
        p.set_background("#111d2b")
        self.body_actors = []
        scale = 60.0
        if env is not None:
            width, length, depth = env.config.phantom_size_mm
            box = pv.Box(bounds=(-width / 2, width / 2, -length / 2, length / 2, 0, depth))
            self.body_actors = [
                p.add_mesh(box, style="wireframe", color="#6c99a8", line_width=1, opacity=.5, reset_camera=False),
                p.add_mesh(pv.wrap(env.mesh), color="#ef6c78", opacity=.12, reset_camera=False),
            ]
            scale = max(scale, length * .6, width * .6, depth * 1.2)
        p.add_text(f"{title}\ncolour: when recorded (dark = early, yellow = late)", position="upper_left",
                   font_size=10, color="#d6e2ee")
        p.add_text("", position="lower_left", font_size=10, color="#b8c9da", name="count")
        # +z points into the tissue, so the camera looks from above the surface with -z up.
        p.camera_position = [(125, -175, -135), (0, 0, 10), (0, 0, -1)]
        p.camera.parallel_projection = True
        p.camera.parallel_scale = scale
        self.chunks, self.steps = [], []
        self._framed = env is not None

    @property
    def points(self):
        """All accumulated points, (N, 3)."""
        return np.vstack(self.chunks) if self.chunks else np.empty((0, 3))

    @property
    def order(self):
        """Step at which each accumulated point was added, (N,)."""
        return np.concatenate(self.steps) if self.steps else np.empty(0)

    def add(self, points, step):
        """Collect one step's (N, 3) points; an exact repeat of the last chunk is skipped.
        Returns True if the points were added. Display them with draw_points()."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        last = self.chunks[-1] if self.chunks else None
        if len(points) == 0 or (last is not None and last.shape == points.shape and np.allclose(last, points)):
            return False
        self.chunks.append(points)
        self.steps.append(np.full(len(points), step))
        return True

    def draw_points(self, points=None, order=None, total=None):
        """Display points coloured by order (dark = early, yellow = late); default: all accumulated.
        total: size of the cloud the points were taken from, e.g. when showing a fit's subset."""
        if points is None:
            points, order = self.points, self.order
        p = self.plotter
        if len(points) == 0:
            p.remove_actor("cloud", render=False)
        else:
            cloud = self.pv.PolyData(np.asarray(points, dtype=float))
            cloud["order"] = np.asarray(order, dtype=float)
            p.add_mesh(cloud, scalars="order", cmap="viridis", point_size=3, show_scalar_bar=False,
                       lighting=False, name="cloud", reset_camera=False, render=False)
            if not self._framed:  # no phantom to frame: centre the camera on the first data
                p.camera.focal_point = cloud.center
                p.camera.position = np.add(cloud.center, (125, -175, -145))
                self._framed = True
        label = (f"{len(points):,} points from {len(self.chunks)} frames" if total is None
                 else f"{len(points):,} points used by the fit (of {total:,} recorded)")
        p.add_text(label, position="lower_left", font_size=10, color="#b8c9da", name="count", render=False)
        p.render()

    def set_fit(self, cylinder, label="fit", curve=None, weights=None):
        """Draw a fitted bent cylinder (object with centreline(n) and radius), or remove it with None.
        curve: optional (N, 3) centreline samples to build the tube along (e.g. already moved into
        this window's frame); weights: their weights in a subsequent fit, shaded dim (0) to bright."""
        p = self.plotter
        if cylinder is None:
            for name in ("fit_tube", "fit_line"):
                p.remove_actor(name, render=False)
            p.add_text(f"{label}: needs more points along the vessel", position="upper_right",
                       font_size=10, color="#f0c05a", name="fit_text", render=False)
        else:
            weighted = curve is not None and weights is not None and len(curve) == len(weights)
            line = self.pv.lines_from_points(np.asarray(curve) if curve is not None else cylinder.centreline(200))
            if weighted:
                line["weight"] = np.asarray(weights, dtype=float)
                style = dict(scalars="weight", cmap=["#4a3d22", "#ffd36b"], show_scalar_bar=False,
                             clim=(0, max(line["weight"].max(), 1e-9)))
            else:
                style = dict(color="#f0c05a")
            p.add_mesh(line.tube(radius=cylinder.radius, n_sides=48), opacity=.35, smooth_shading=True,
                       name="fit_tube", reset_camera=False, render=False, **style)
            p.add_mesh(line, color="#f0c05a", line_width=3, name="fit_line", reset_camera=False, render=False)
            p.add_text(f"{label}: radius {cylinder.radius:.2f} mm\nrms {cylinder.rms:.3f} mm",
                       position="upper_right", font_size=10, color="#f0c05a", name="fit_text", render=False)
        p.render()

    def set_help(self, text):
        self.plotter.add_text(text, position=(.02, .80), viewport=True, font_size=10,
                              color="#b8c9da", name="help", render=False)

    def set_probe(self, pose, fov_mm=(40.0, 40.0)):
        """Draw a probe pose (4x4, mm) as its image plane and x/y/z axes (red/green/blue)."""
        p, pose = self.plotter, np.asarray(pose, dtype=float)
        fw, fd = fov_mm
        corners = np.array([[-fw / 2, 0, 0], [fw / 2, 0, 0], [fw / 2, 0, fd], [-fw / 2, 0, fd]])
        p.add_mesh(self.pv.PolyData(corners @ pose[:3, :3].T + pose[:3, 3], [4, 0, 1, 2, 3]),
                   color="#e4b775", opacity=.2, show_edges=True, name="probe_plane", reset_camera=False, render=False)
        for i, (label, color) in enumerate(zip("xyz", ["#fc777a", "#8adb98", "#79b4ff"])):
            p.add_mesh(self.pv.Arrow(start=pose[:3, 3], direction=pose[:3, i], scale=15), color=color,
                       name="probe_axis_" + label, reset_camera=False, render=False)
        p.render()

    def set_plane(self, corners, text=""):
        """Draw a plane patch from (4, 3) corners with a label, or remove it with None."""
        p = self.plotter
        if corners is None:
            p.remove_actor("plane", render=False)
        else:
            patch = self.pv.PolyData(np.asarray(corners, dtype=float), [4, 0, 1, 2, 3])
            p.add_mesh(patch, color="#79b4ff", opacity=.25, show_edges=True, edge_color="#79b4ff",
                       name="plane", reset_camera=False, render=False)
        p.add_text(text, position="lower_right", font_size=10, color="#79b4ff", name="plane_text", render=False)
        p.render()

    def set_body_offset(self, offset):
        shift = np.eye(4)
        shift[:2, 3] = offset
        for actor in self.body_actors:
            actor.user_matrix = shift
        self.plotter.render()
