#!/usr/bin/env python3
"""Scan absolute Roll/Yaw at X=Y=Pitch=0 and map the Phantom 1 longitudinal-view region.

Provides the evidence behind the success thresholds and the Roll limit in
configs/tasks/longitudinal_phantom1.json. Needs matplotlib for the heatmap.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from view_transition_rl.envs.longitudinal_view import load_task_config
from view_transition_rl.envs.metrics import is_longitudinal, longitudinal_metrics
from view_transition_rl.simulation import PhantomEnv
from view_transition_rl.simulation.geometry import pose_from_euler


def scan(rolls, yaws, success):
    env = PhantomEnv(phantom="reference")
    shape = (len(rolls), len(yaws))
    run, thickness = np.full(shape, np.nan), np.full(shape, np.nan)
    for i, roll in enumerate(rolls):
        for j, yaw in enumerate(yaws):
            if env.set_pose(pose_from_euler(0, 0, roll=np.deg2rad(roll), yaw=np.deg2rad(yaw))).accepted:
                metrics = longitudinal_metrics(env.get_mask(), env.config.fov_mm)
                run[i, j], thickness[i, j] = metrics["column_run"], metrics["thickness_mm"]
    ok = (run >= success["min_column_run"]) & (thickness >= success["min_thickness_mm"])
    return run, thickness, ok


def region(rolls, yaws, ok):
    cells = np.argwhere(ok)
    if not len(cells):
        return None
    return {"roll_deg": [float(rolls[cells[:, 0]].min()), float(rolls[cells[:, 0]].max())],
            "yaw_deg": [float(yaws[cells[:, 1]].min()), float(yaws[cells[:, 1]].max())], "cells": int(len(cells))}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default="reports/tasks/longitudinal_phantom1")
    parser.add_argument("--roll-step", type=float, default=1.0)
    parser.add_argument("--yaw-step", type=float, default=2.0)
    args = parser.parse_args()
    config = load_task_config()
    rolls = np.arange(-30, 30 + 1e-9, args.roll_step)
    yaws = np.arange(-180, 180, args.yaw_step)
    run, thickness, ok = scan(rolls, yaws, config["success"])
    positive = np.abs(((yaws + 180) % 360) - 180 - 90) <= 45
    negative = np.abs(((yaws + 180) % 360) - 180 + 90) <= 45
    best = np.unravel_index(np.nanargmax(np.where(ok, thickness, -1)), ok.shape)
    zero_roll = int(np.argmin(np.abs(rolls)))
    summary = {
        "phantom": "reference", "fixed": {"x_mm": 0, "y_mm": 0, "pitch_deg": 0},
        "grid_step_deg": {"roll": args.roll_step, "yaw": args.yaw_step},
        "success": config["success"],
        "success_region_yaw_pos": region(rolls, yaws, ok & positive[None]),
        "success_region_yaw_neg": region(rolls, yaws, ok & negative[None]),
        "best_pose_deg": {"roll": float(rolls[best[0]]), "yaw": float(yaws[best[1]]),
                          "thickness_mm": float(thickness[best])},
        "max_abs_roll_with_success_deg": float(np.abs(rolls[ok.any(axis=1)]).max()),
        "max_column_run_at_zero_roll": float(np.nanmax(run[zero_roll])),
        "configured_max_roll_deg": config["max_roll_deg"],
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(output / "scan.npz", rolls=rolls, yaws=yaws, column_run=run, thickness_mm=thickness, success=ok)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    extent = [yaws[0] - args.yaw_step / 2, yaws[-1] + args.yaw_step / 2,
              rolls[0] - args.roll_step / 2, rolls[-1] + args.roll_step / 2]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for ax, data, title, unit in [(axes[0], run, "Longest contiguous column run", "fraction of width"),
                                  (axes[1], thickness, "Mean vessel thickness", "mm")]:
        image = ax.imshow(data, origin="lower", aspect="auto", extent=extent, cmap="viridis")
        ax.contour(yaws, rolls, ok.astype(float), levels=[0.5], colors="white", linewidths=1.2)
        for limit in (-config["max_roll_deg"], config["max_roll_deg"]):
            ax.axhline(limit, color="tab:red", linestyle="--", linewidth=0.8)
        ax.plot(0, 0, marker="x", color="tab:red")
        ax.set(title=title, xlabel="Yaw (deg)", ylabel="Roll (deg)")
        fig.colorbar(image, ax=ax, label=unit)
    fig.suptitle("Phantom 1, X=Y=Pitch=0: white = longitudinal success, red dashed = Roll limit, x = start")
    fig.savefig(output / "roll_yaw_scan.png", dpi=130)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
