"""Compatibility forwarding for simulation environment APIs."""

from view_transition_rl.simulation.environment import (
    PhantomConfig, PhantomEnv, StepResult, asset_dir, load_config,
)

__all__ = ["PhantomConfig", "PhantomEnv", "StepResult", "asset_dir", "load_config"]
