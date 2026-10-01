# Policies and algorithms (reserved)

This module is reserved for custom policy networks, action selection, and RL framework adapters. Policies interact through task observation and action interfaces and should not depend on the GUI or private simulator meshes.

Current training uses Stable-Baselines3 DQN and its built-in `MultiInputPolicy` directly from the training module. Saved models are included under `outputs/training/`; no custom policy implementation is currently needed here.
