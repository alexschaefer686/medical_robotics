# Reinforcement learning task environments

- `metrics.py`: pure NumPy mask metrics for longitudinal views: longest continuous column coverage and mean lumen thickness.
- `longitudinal_view.py`: `LongitudinalViewEnv` (`ViewTransition/LongitudinalPhantom1-v0`), transitioning from transverse to longitudinal on phantom 1.

Task configuration defines observations, actions, rewards, and termination. Keep training semantics out of the underlying simulator.
