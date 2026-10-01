# Project architecture

The project supports reinforcement learning for view transitions. The geometric simulator provides data and pose control; the task environment adds RL semantics.

## Module boundaries

```text
training -> SB3 DQN -> envs -> simulation
evaluation -> saved SB3 policy + envs
agents -> reserved for future custom policy implementations
```

- **simulation:** closed vessel meshes, phantom/probe geometry, constrained pose updates, ideal segmentation masks, and a debugging GUI. `PhantomEnv.step()` reports whether a motion was accepted; it defines no rewards, episodes, or training behavior.
- **envs:** task goals, observations, actions, rewards, resets, and termination. The phantom 1 longitudinal task is implemented. Gymnasium is an optional `rl` dependency; importing `view_transition_rl.envs` does not load it, while `register_envs()` registers the environment ID.
- **agents:** reserved for custom networks and algorithm adapters. Policies should use task observations and actions, without depending on GUI objects or private simulator meshes. Current training uses SB3's built-in policy directly.
- **training:** combines the task and policy, managing configuration, seeds, logs, and checkpoints through `train_dqn.py`. Rewards belong in `envs`; geometry belongs in `simulation`.
- **evaluation:** loads policies and measures success and action counts under fixed task conditions. `evaluate_policy.py` differs from `simulation.evaluate`, which compares simulated masks against acquisition masks.

## Data, configuration, and outputs

Simulator assets live only in `src/view_transition_rl/simulation/assets`, are located through `importlib.resources`, and are packaged in the wheel. Original provenance and checksums are retained.

`configs/simulation`, `configs/tasks`, and `configs/training` contain implemented configurations. Runtime outputs live under `outputs/<module-or-experiment>/`; retained analyses live under `reports/<module>/`. The shared copy includes trained models, logs, checkpoints, and evaluations.

One-time resource import belongs in `scripts/simulation`; installation checks belong in `scripts/verification`. Runtime modules do not import these maintenance scripts.

## Installation and compatibility

The distribution is named `view-transition-rl` and the Python namespace is `view_transition_rl`. Root and reserved RL package imports stay lightweight and do not eagerly load VTK or training frameworks.

`view-transition-rl sim ...` forwards to the simulator CLI. Put simulator options after `sim` and before its subcommand, for example `view-transition-rl sim --config configs/simulation/default.json gui`. RL training and evaluation use their Python module entrypoints.

`vessel_phantom` provides explicit import forwarding and legacy CLI adapters; it does not duplicate assets. Existing `python -m vessel_phantom ...` and `vessel-phantom ...` commands remain supported.

For an existing installation of the old distribution, uninstall the old `vessel-phantom` package metadata before installing this project in editable mode. Dependencies such as NumPy, VTK, and PyVista need not be removed. For this shared copy, create a fresh virtual environment; none is bundled.
