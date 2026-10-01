"""GUI regression tests use complete VTK key sequences, including CharEvent.

Run explicitly (requires the GUI extra and an offscreen OpenGL backend):
    PHANTOM_TEST_GUI=1 python -m unittest discover -s tests -p test_gui_keyboard.py -v
"""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from view_transition_rl.simulation import PhantomEnv


@unittest.skipUnless(os.environ.get("PHANTOM_TEST_GUI") == "1", "set PHANTOM_TEST_GUI=1 for GUI regression tests")
class TestKeyboard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from view_transition_rl.simulation.gui import PhantomViewer

        cls.directory = tempfile.TemporaryDirectory()
        cls.viewer = PhantomViewer(PhantomEnv(), off_screen=True, export_dir=cls.directory.name)
        cls.viewer.plotter.show(auto_close=False, interactive=False)
        cls.interactor = cls.viewer.plotter.iren.interactor
        cls.exit_events = []
        # Record exit attempts instead of letting a regression terminate tests.
        cls.exit_observer = cls.interactor.AddObserver(
            "ExitEvent", lambda *_: cls.exit_events.append(cls.interactor.GetKeySym()))

    @classmethod
    def tearDownClass(cls):
        cls.interactor.RemoveObserver(cls.exit_observer)
        cls.viewer.close()
        cls.directory.cleanup()

    def setUp(self):
        self.viewer.reset()
        self.exit_events.clear()

    def press(self, key, repeat=0, mouse=(400, 450)):
        self.interactor.SetEventPosition(*mouse)
        code = key if len(key) == 1 else "\0"
        self.interactor.SetKeyEventInformation(0, int(key.isupper()), code, repeat, key)
        for event in ["KeyPressEvent", "CharEvent", "KeyReleaseEvent"]:
            self.interactor.InvokeEvent(event)

    def scene_signature(self):
        signature = []
        for index, name in [(0, "vessel"), (0, "outline"), (0, "surface"), (1, "mask")]:
            actor = self.viewer.plotter.renderers[index].actors[name]
            prop = actor.GetProperty()
            signature.append((actor.GetAddressAsString(""), actor.GetVisibility(),
                              prop.GetRepresentation(), prop.GetOpacity()))
        return signature

    def pose(self):
        return np.array(self.viewer.env.get_state()["phantom_T_probe_mm"])

    def test_phantom_switch(self):
        viewer = self.viewer
        try:
            viewer.pending = np.eye(4)
            viewer.phantom_button.InvokeEvent("StateChangedEvent")
            self.assertEqual(viewer.phantom_button.GetRepresentation().GetState(), 1)
            self.assertIsNone(viewer.pending)
            self.assertEqual(viewer.env.phantom_id, "vessel_no_hole")
            self.assertEqual(viewer.widgets["y"].GetRepresentation().GetMaximumValue(), 95)
            np.testing.assert_array_equal(viewer.mask_grid.point_data["mask"], viewer.env.get_mask()[::-1].ravel())
            self.assertGreater(viewer.env.get_mask().sum(), 0)
            viewer.phantom_button.InvokeEvent("StateChangedEvent")
            self.assertEqual(viewer.env.phantom_id, "curved_arc")
            self.assertEqual(viewer.phantom_button.GetRepresentation().GetState(), 2)
            self.assertEqual(viewer.widgets["x"].GetRepresentation().GetMaximumValue(), 50)
            np.testing.assert_array_equal(viewer.env.get_state()["phantom_T_probe_mm"], np.eye(4))
            np.testing.assert_array_equal(viewer.mask_grid.point_data["mask"], viewer.env.get_mask()[::-1].ravel())
            viewer.phantom_button.InvokeEvent("StateChangedEvent")
            self.assertEqual(viewer.env.phantom_id, "reference")
            self.assertEqual(viewer.widgets["y"].GetRepresentation().GetMaximumValue(), 50)
            original = viewer.env
            viewer.env.config_overrides = {"phantom_size_mm": [70, 100, 40]}
            viewer.phantom_button.GetRepresentation().SetState(1)
            viewer.phantom_button.InvokeEvent("StateChangedEvent")
            self.assertIs(viewer.env, original)
            self.assertIn("Cannot switch", viewer.notice)
            self.assertEqual(viewer.phantom_button.GetRepresentation().GetState(), 0)
        finally:
            viewer.env.config_overrides = {}
            viewer.switch_phantom("reference")

    def test_char_events_do_not_exit_or_change_rendering(self):
        original = self.scene_signature()
        original_mask = self.viewer.env.get_mask()
        cameras = [np.array(renderer.camera.GetPosition()) for renderer in self.viewer.plotter.renderers]
        for mouse in [(400, 450), (1200, 700), (1200, 250)]:
            for key in ["q", "e", "w", "s", "a", "d", "Q", "E", "W", "S", "A", "D", "r", "R"]:
                with self.subTest(key=key, mouse=mouse):
                    self.press(key, mouse=mouse)
                    self.assertEqual(self.exit_events, [])
                    self.assertEqual(self.scene_signature(), original)
                    self.assertIs(self.viewer.plotter.renderers[0].actors["probe"], self.viewer.probe_actor)
                    actor_matrix = self.viewer.probe_actor.GetUserMatrix()
                    self.assertIsNotNone(actor_matrix)
                    for row in range(4):
                        for column in range(4):
                            self.assertAlmostEqual(actor_matrix.GetElement(row, column),
                                                   self.pose()[row, column], places=7)
                    self.assertFalse(self.viewer.plotter._closed)
                    for renderer, camera in zip(self.viewer.plotter.renderers, cameras):
                        np.testing.assert_array_equal(renderer.camera.GetPosition(), camera)
        np.testing.assert_array_equal(self.viewer.env.get_mask(), original_mask)

    def test_single_step_and_held_keys(self):
        from view_transition_rl.simulation.geometry import rotation_xyz

        self.press("e")
        np.testing.assert_allclose(self.pose()[:3, :3], rotation_xyz(rz=np.deg2rad(1)), atol=1e-12)
        for index in range(179):
            self.press("e", repeat=index + 1)
        self.assertEqual(self.exit_events, [])
        np.testing.assert_allclose(self.pose()[:3, :3], rotation_xyz(rz=np.pi), atol=1e-12)
        self.press("R")
        np.testing.assert_array_equal(self.pose(), np.eye(4))
        for key in ["Right", "Up", "Left", "Down"]:
            self.press(key)
        np.testing.assert_allclose(self.pose(), np.eye(4), atol=1e-12)
        self.assertGreater(self.viewer.env.get_mask().sum(), 0)

    def test_limit_rejection_and_export(self):
        for index in range(65):
            self.press("w", repeat=index)
        self.assertFalse(self.viewer.env.get_state()["last_action_accepted"])
        self.assertAlmostEqual(self.viewer.env.get_state()["tilt_deg"], 60)
        self.press("r")
        self.press("P")
        self.assertEqual(self.exit_events, [])
        self.assertTrue(list(Path(self.directory.name).glob("*.png")))
        self.assertTrue(list(Path(self.directory.name).glob("*.json")))


if __name__ == "__main__":
    unittest.main()
