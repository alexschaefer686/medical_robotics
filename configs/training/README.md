# Training configuration

`dqn_longitudinal.json` defines Stable-Baselines3 DQN hyperparameters, the seed, evaluation/checkpoint frequencies, and the output directory. `task_config` points to the task configuration.

```bash
python -m view_transition_rl.training.train_dqn --config configs/training/dqn_longitudinal.json --output outputs/training/new_run
```
