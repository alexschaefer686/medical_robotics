"""Train a stable-baselines3 DQN policy on the Phantom 1 longitudinal-view task.

python -m view_transition_rl.training.train_dqn --config configs/training/dqn_longitudinal.json
"""

import argparse
import json
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from ..envs.longitudinal_view import LongitudinalViewEnv

DQN_KEYS = ("learning_rate", "buffer_size", "learning_starts", "batch_size", "gamma", "train_freq",
            "target_update_interval", "exploration_fraction", "exploration_initial_eps",
            "exploration_final_eps", "device")


def make_env(task_config, monitor_path=None):
    return Monitor(LongitudinalViewEnv(config_path=task_config), monitor_path,
                   info_keywords=("success", "roll_deg", "yaw_deg"))


def train(config, output_dir=None):
    output = Path(output_dir or config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    env = make_env(config["task_config"], str(output / "train"))
    eval_env = make_env(config["task_config"], str(output / "eval"))
    model = DQN("MultiInputPolicy", env, seed=config["seed"], verbose=1,
                tensorboard_log=str(output / "tensorboard"), **{k: config[k] for k in DQN_KEYS})
    callbacks = [
        EvalCallback(eval_env, best_model_save_path=str(output), log_path=str(output / "eval_logs"),
                     eval_freq=config["eval_freq"], n_eval_episodes=config["n_eval_episodes"], deterministic=True),
        CheckpointCallback(config["checkpoint_freq"], str(output / "checkpoints"), name_prefix="dqn"),
    ]
    model.learn(total_timesteps=config["total_timesteps"], callback=callbacks)
    model.save(output / "final_model")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="Training JSON, e.g. configs/training/dqn_longitudinal.json")
    parser.add_argument("--output", help="Override output_dir from the config")
    parser.add_argument("--timesteps", type=int, help="Override total_timesteps")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text())
    if args.timesteps is not None:
        config["total_timesteps"] = args.timesteps
    print(f"Saved run to {train(config, args.output)}")


if __name__ == "__main__":
    main()
