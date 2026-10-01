# Policy evaluation

`evaluate_policy.py` loads a DQN model and performs deterministic rollouts from the fixed initial pose, exporting success rate, step counts, a trajectory CSV, and the final mask.

`render_video.py` renders `trajectory.csv` using the simulator GUI scene offscreen and produces an MP4 replay.

STL-slice versus reference-mask Dice/IoU comparisons remain the responsibility of `view_transition_rl.simulation.evaluate`.
