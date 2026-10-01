"""Roll out a trained DQN policy on the longitudinal-view task and save the trajectory.

python -m view_transition_rl.evaluation.evaluate_policy --model outputs/training/dqn_longitudinal/best_model.zip
"""

import argparse
import csv
import json
from pathlib import Path

from PIL import Image
from stable_baselines3 import DQN

from ..envs.longitudinal_view import ACTION_NAMES, LongitudinalViewEnv


def evaluate(model_path, task_config=None, episodes=1, output_dir=None):
    env = LongitudinalViewEnv(config_path=task_config)
    model = DQN.load(model_path, device="cpu")
    episodes_summary, trajectory = [], []
    for episode in range(episodes):
        obs, info = env.reset(seed=episode)
        total, done = 0.0, False
        trajectory.append({"episode": episode, "step": 0, "action": "", "reward": 0.0, **info})
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total += reward
            done = terminated or truncated
            trajectory.append({"episode": episode, "step": info["steps"], "action": ACTION_NAMES[int(action)],
                               "reward": reward, **info})
        episodes_summary.append({"success": info["success"], "steps": info["steps"], "return": total,
                                 "final_roll_deg": info["roll_deg"], "final_yaw_deg": info["yaw_deg"],
                                 "column_run": info["column_run"], "thickness_mm": info["thickness_mm"]})
    summary = {"model": str(model_path), "episodes": episodes,
               "success_rate": sum(e["success"] for e in episodes_summary) / episodes,
               "mean_steps": sum(e["steps"] for e in episodes_summary) / episodes, "per_episode": episodes_summary}
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        with (output / "trajectory.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(trajectory[0]))
            writer.writeheader()
            writer.writerows(trajectory)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        Image.fromarray(env.current_mask()).save(output / "final_mask.png")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--task-config", help="Task JSON; defaults to the built-in Phantom 1 task")
    parser.add_argument("--episodes", type=int, default=1, help="The task is deterministic; one episode suffices")
    parser.add_argument("--output", default="outputs/evaluation/dqn_longitudinal")
    args = parser.parse_args(argv)
    summary = evaluate(args.model, args.task_config, args.episodes, args.output)
    print(json.dumps({k: summary[k] for k in ("success_rate", "mean_steps")}, indent=2))
    print(json.dumps(summary["per_episode"][-1], indent=2))


if __name__ == "__main__":
    main()
