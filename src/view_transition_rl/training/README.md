# Training

`train_dqn.py` trains the longitudinal-view task with Stable-Baselines3 DQN (`MultiInputPolicy`). Configuration is in `configs/training/dqn_longitudinal.json`; outputs go under `outputs/training/`.

Rewards belong in `envs`; geometric calculations belong in `simulation`. Use `--output outputs/training/new_run` to preserve the included run.
