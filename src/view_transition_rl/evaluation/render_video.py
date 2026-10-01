"""Render an evaluated trajectory as an MP4 using the simulator's offscreen GUI scene.

python -m view_transition_rl.evaluation.render_video \
    --trajectory outputs/evaluation/dqn_longitudinal_final_model/trajectory.csv

Each frame sets the replayed pose on the simulator and re-slices the vessel, so
the 3D scene, live mask and pose sliders are exactly what the GUI would show.
Roll/Yaw are linearly interpolated between steps (Yaw along the shorter arc).
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from ..envs.longitudinal_view import load_task_config
from ..envs.metrics import is_longitudinal, longitudinal_metrics
from ..simulation import PhantomEnv
from ..simulation.geometry import pose_from_euler


def read_trajectory(path, episode=0):
    with Path(path).open() as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["episode"]) == episode]
    if not rows:
        raise ValueError(f"No rows for episode {episode} in {path}")
    return rows


def keyframes(rows, substeps):
    """(roll_deg, yaw_deg, row) per video frame; the row is the step being executed."""
    frames = [(float(rows[0]["roll_deg"]), float(rows[0]["yaw_deg"]), rows[0])]
    for previous, row in zip(rows, rows[1:]):
        roll0, yaw0 = float(previous["roll_deg"]), float(previous["yaw_deg"])
        roll1, yaw1 = float(row["roll_deg"]), float(row["yaw_deg"])
        d_yaw = (yaw1 - yaw0 + 180.0) % 360.0 - 180.0
        for k in range(1, substeps + 1):
            t = k / substeps
            frames.append((roll0 + t * (roll1 - roll0), yaw0 + t * d_yaw, row))
    return frames


def render_video(trajectory, output, task_config=None, episode=0, fps=15, substeps=6, hold_s=1.5):
    import imageio.v2 as imageio

    from ..simulation.gui import PhantomViewer

    config = load_task_config(task_config)
    init = config["initial_pose"]
    rows = read_trajectory(trajectory, episode)
    env = PhantomEnv(phantom=config["phantom"])
    viewer = PhantomViewer(env, off_screen=True)
    plotter = viewer.plotter
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = keyframes(rows, substeps)
    hold = int(round(hold_s * fps))
    last_step = int(rows[-1]["step"])
    try:
        with imageio.get_writer(output, fps=fps, codec="libx264", quality=8, macro_block_size=8) as writer:
            for index, (roll, yaw, row) in enumerate(frames):
                pose = pose_from_euler(init["x_mm"], init["y_mm"], roll=np.deg2rad(roll),
                                       pitch=np.deg2rad(init["pitch_deg"]), yaw=np.deg2rad(yaw))
                result = env.set_pose(pose)
                if not result.accepted:
                    raise ValueError(f"Replayed pose rejected by the simulator: {result.reason}")
                metrics = longitudinal_metrics(env.get_mask(), env.config.fov_mm)
                success = is_longitudinal(metrics, **config["success"])
                step = int(row["step"])
                action = row["action"] or "start"
                if row["rejected"] == "True":
                    action += " (rejected: roll limit)"
                viewer.notice = f"Step {step}/{last_step}   action: {action}"
                viewer.refresh()
                plotter.subplot(0, 1)
                plotter.add_text(
                    f"Roll {roll:+6.1f} deg   Yaw {yaw:+7.1f} deg\n"
                    f"Column run {metrics['column_run']:.2f} (>= {config['success']['min_column_run']:.2f})   "
                    f"Thickness {metrics['thickness_mm']:.2f} mm (>= {config['success']['min_thickness_mm']:g})",
                    position=(.105, .215), viewport=True, font_size=11, color="#f0f5fc",
                    name="task_metrics", render=False)
                plotter.add_text("LONGITUDINAL VIEW REACHED" if success else "Searching for longitudinal view...",
                                 position=(.105, .155), viewport=True, font_size=15,
                                 color="#8adb98" if success else "#e4b775", name="task_status", render=False)
                plotter.subplot(0, 0)
                plotter.render()
                image = plotter.screenshot(return_img=True)
                repeat = hold if index in (0, len(frames) - 1) else 1
                for _ in range(repeat):
                    writer.append_data(image)
    finally:
        viewer.close()
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trajectory", required=True, help="trajectory.csv from evaluate_policy")
    parser.add_argument("--output", help="MP4 path; defaults to trajectory.mp4 next to the CSV")
    parser.add_argument("--task-config", help="Task JSON; defaults to the built-in Phantom 1 task")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--substeps", type=int, default=6, help="Interpolated frames per action")
    parser.add_argument("--hold", type=float, default=1.5, help="Seconds to hold the first and last frame")
    args = parser.parse_args(argv)
    output = args.output or Path(args.trajectory).with_name("trajectory.mp4")
    print(render_video(args.trajectory, output, args.task_config, args.episode, args.fps, args.substeps, args.hold))


if __name__ == "__main__":
    main()
