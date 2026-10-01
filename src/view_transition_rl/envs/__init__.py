"""Task-specific RL environments built on view_transition_rl.simulation.

Importing this package stays lightweight (no Gymnasium, VTK or torch). Call
register_envs() before gymnasium.make(); Gymnasium comes from the "rl" extra.
"""

ENV_IDS = {"ViewTransition/LongitudinalPhantom1-v0": "view_transition_rl.envs.longitudinal_view:LongitudinalViewEnv"}


def register_envs():
    import gymnasium

    for env_id, entry_point in ENV_IDS.items():
        if env_id not in gymnasium.registry:
            gymnasium.register(id=env_id, entry_point=entry_point)
    return tuple(ENV_IDS)
