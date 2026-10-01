import json
import unittest
from pathlib import Path

import numpy as np

from view_transition_rl.envs.metrics import is_longitudinal, longitudinal_metrics

try:
    import gymnasium
except ImportError:
    gymnasium = None

ROOT = Path(__file__).resolve().parents[2]


class TestLongitudinalMetrics(unittest.TestCase):
    def test_empty_thin_split_and_full_band(self):
        mask = np.zeros((480, 480), np.uint8)
        self.assertFalse(is_longitudinal(longitudinal_metrics(mask)))
        thin = mask.copy()
        thin[200:220, :] = 255  # 20 px = 1.67 mm: grazes the wall
        metrics = longitudinal_metrics(thin)
        self.assertEqual(metrics["column_run"], 1.0)
        self.assertFalse(is_longitudinal(metrics))
        split = mask.copy()
        split[150:220, :230] = 255
        split[150:220, 250:] = 255  # two separate pieces covering 96% of the width
        metrics = longitudinal_metrics(split)
        self.assertAlmostEqual(metrics["coverage"], 460 / 480)
        self.assertFalse(is_longitudinal(metrics))
        band = mask.copy()
        band[150:220, 10:] = 255  # 70 px = 5.83 mm over 98% of the width
        metrics = longitudinal_metrics(band)
        self.assertAlmostEqual(metrics["thickness_mm"], 70 * 40 / 480)
        self.assertTrue(is_longitudinal(metrics))


@unittest.skipIf(gymnasium is None, "requires the rl extra (gymnasium)")
class TestLongitudinalViewEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from view_transition_rl.envs.longitudinal_view import LongitudinalViewEnv
        cls.env_class = LongitudinalViewEnv

    def test_config_file_matches_defaults(self):
        from view_transition_rl.envs.longitudinal_view import DEFAULT_TASK_CONFIG, load_task_config
        path = ROOT / "configs/tasks/longitudinal_phantom1.json"
        self.assertEqual(json.loads(path.read_text()), DEFAULT_TASK_CONFIG)
        self.assertEqual(load_task_config(path), DEFAULT_TASK_CONFIG)
        with self.assertRaises(ValueError):
            load_task_config(unknown=1)

    def test_reset_is_zero_pose_cross_section(self):
        env = self.env_class()
        obs, info = env.reset(seed=0)
        np.testing.assert_allclose(env.current_pose(), np.eye(4), atol=1e-12)
        self.assertFalse(info["success"])
        self.assertAlmostEqual(info["column_run"], 0.1125, places=3)
        self.assertEqual(obs["mask"].shape, (1, 84, 84))
        self.assertTrue(env.observation_space.contains(obs))

    def test_scripted_path_succeeds(self):
        env = self.env_class()
        env.reset()
        for step, action in enumerate([3] * 18 + [0] * 5, 1):
            _, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break
        self.assertTrue(terminated and info["success"])
        self.assertFalse(truncated)
        self.assertLessEqual(step, 23)
        self.assertGreater(reward, 10)
        self.assertEqual((info["roll_deg"], info["yaw_deg"]), (-4.0, 90.0))

    def test_roll_limit_rejects_without_moving(self):
        env = self.env_class()
        env.reset()
        for _ in range(10):
            _, _, _, _, info = env.step(1)
            self.assertFalse(info["rejected"])
        self.assertEqual(info["roll_deg"], 20.0)
        pose = env.current_pose()
        _, reward, _, _, info = env.step(1)
        self.assertTrue(info["rejected"])
        self.assertEqual(info["roll_deg"], 20.0)
        np.testing.assert_array_equal(env.current_pose(), pose)
        self.assertAlmostEqual(reward, -0.25)

    def test_pitch_stays_zero_and_yaw_wraps(self):
        from view_transition_rl.simulation.geometry import euler_from_pose
        env = self.env_class()
        env.reset()
        rng = np.random.default_rng(1)
        for action in rng.integers(0, 4, 60):
            env.step(int(action))
            roll, pitch, yaw = np.rad2deg(euler_from_pose(env.current_pose()))
            info_roll, info_yaw = env.angles_deg()
            self.assertAlmostEqual(pitch, 0, places=9)
            self.assertAlmostEqual(roll, info_roll, places=9)
            self.assertAlmostEqual((yaw - info_yaw + 180) % 360 - 180, 0, places=9)
            self.assertTrue(-180 <= info_yaw < 180)

    def test_truncation_and_env_checker(self):
        from gymnasium.utils.env_checker import check_env
        env = self.env_class(config={"max_episode_steps": 3})
        env.reset()
        results = [env.step(2) for _ in range(3)]
        self.assertEqual([r[3] for r in results], [False, False, True])
        check_env(self.env_class(), skip_render_check=True)

    def test_registered_id(self):
        from view_transition_rl.envs import register_envs
        self.assertIn("ViewTransition/LongitudinalPhantom1-v0", register_envs())
        env = gymnasium.make("ViewTransition/LongitudinalPhantom1-v0")
        self.assertIsInstance(env.unwrapped, self.env_class)


if __name__ == "__main__":
    unittest.main()
