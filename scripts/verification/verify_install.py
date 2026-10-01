#!/usr/bin/env python3
"""Test a wheel from a temporary directory, denying access to source trees."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--gui", action="store_true", help="Also exercise desktop controls and screenshot")
    parser.add_argument("--output", type=Path, default=Path("reports/simulation/install_verification"))
    args = parser.parse_args()
    wheel = args.wheel.resolve(strict=True)
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="vessel-phantom-install-") as directory:
        temporary = Path(directory)
        installed = temporary / "installed"
        subprocess.run([sys.executable, "-m", "pip", "--disable-pip-version-check", "install",
                        "--no-index", "--no-deps", "--no-cache-dir", "--target", str(installed), str(wheel)], check=True)
        code = r'''
import os, sys, json
from pathlib import Path
blocked_roots = [Path(p).resolve() for p in json.loads(os.environ["PHANTOM_DENY_ROOTS"])]
# The invoking Python may live in the project's .venv. Its dependencies are
# allowed; project source and assets remain forbidden outside the wheel.
runtime_root = Path(sys.prefix).resolve()
audited_reads = 0
def audit(event, args):
    global audited_reads
    if event not in {"open", "os.listdir", "os.scandir"} or not args or not isinstance(args[0], (str, bytes)):
        return
    path = Path(os.fsdecode(args[0])).absolute()
    runtime_dependency = path == runtime_root or runtime_root in path.parents
    if not runtime_dependency and any(path == root or root in path.parents for root in blocked_roots):
        raise PermissionError("Source-tree access forbidden during standalone verification: " + str(path))
    audited_reads += 1
sys.addaudithook(audit)
import numpy as np
import vessel_phantom
import view_transition_rl.simulation as simulation
from view_transition_rl.simulation import PhantomEnv
from view_transition_rl.simulation.evaluate import evaluate
from view_transition_rl.simulation.geometry import pose_from_euler
assert Path(vessel_phantom.__file__).is_relative_to(Path(os.environ["PYTHONPATH"]))
assert Path(simulation.__file__).is_relative_to(Path(os.environ["PYTHONPATH"]))
assert vessel_phantom.PhantomEnv is PhantomEnv
second = PhantomEnv(phantom="vessel_no_hole")
assert second.assets.is_relative_to(Path(os.environ["PYTHONPATH"]))
assert second.get_mask().shape == (480, 480) and second.get_mask().any()
assert second.get_state()["phantom_id"] == "vessel_no_hole"
second.export("second_phantom")
third = PhantomEnv(phantom="curved_arc")
assert third.assets.is_relative_to(Path(os.environ["PYTHONPATH"]))
assert third.get_mask().shape == (480, 480) and third.get_mask().any()
assert third.set_pose(third.calibration["example_poses"]["longitudinal"]).accepted
assert third.get_mask().any()
third.export("third_phantom")
env = PhantomEnv()
assert env.assets.is_relative_to(Path(os.environ["PYTHONPATH"]))
assert env.set_pose(pose_from_euler(0, 0, .2, -.1, .3)).accepted
assert env.step([1, -1, .02, -.01, .03]).accepted
assert env.get_mask().shape == (480, 480)
env.export("sample")
for i in range(190):
    assert env.replay_frame(i).accepted
summary = evaluate("evaluation", env)
assert summary["frames"] == 190 and summary["dice"]["mean"] > .9
# Exercise module and console entry points with the same audit hook active.
import contextlib, io, runpy
from importlib.metadata import distribution
def module_info(module, arguments):
    previous = sys.argv
    sys.argv = [module, *arguments]
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            runpy.run_module(module, run_name="__main__")
    finally:
        sys.argv = previous
    return json.loads(output.getvalue())
assert module_info("view_transition_rl", ["sim", "info"]) == module_info("vessel_phantom", ["info"])
entrypoints = {ep.name: ep for ep in distribution("view-transition-rl").entry_points if ep.group == "console_scripts"}
for name, arguments in [("view-transition-rl", ["sim", "sample", "--output", "console_new"]),
                        ("vessel-phantom", ["sample", "--output", "console_old"])]:
    entrypoints[name].load()(arguments)
assert Path("console_new.png").read_bytes() == Path("console_old.png").read_bytes()
result = {"passed": True, "source_access_blocked": [str(p) for p in blocked_roots],
          "runtime_dependencies_allowed": str(runtime_root),
          "new_and_legacy_entrypoints": True, "single_packaged_asset_location": True,
          "all_190_replay_frames": True, "evaluation_frames": summary["frames"],
          "dice_mean": summary["dice"]["mean"], "gui_tested": False, "all_three_phantoms_loaded": True}
if os.environ.get("PHANTOM_VERIFY_GUI") == "1":
    from view_transition_rl.simulation.gui import PhantomViewer
    env.reset()
    viewer = PhantomViewer(env, off_screen=True, export_dir="captures")
    probe_bounds = viewer.probe_mesh.bounds
    np.testing.assert_allclose(probe_bounds, [-23, 24, -14, 14, -50, 0], atol=1)
    assert viewer.probe_actor.GetUserMatrix() is not None
    old_pose = np.array(env.get_state()["phantom_T_probe_mm"])
    widget = viewer.widgets["x"]
    widget.GetRepresentation().SetValue(old_pose[0, 3] + 1)
    widget.InvokeEvent("InteractionEvent")
    viewer.flush_pending()
    assert abs(env.get_state()["phantom_T_probe_mm"][0][3] - old_pose[0, 3] - 1) < 1e-8
    viewer.move([0, .5, .02, 0, 0])
    assert env.get_state()["last_action_accepted"]
    before = env.get_mask()
    viewer.move([1000, 0, 0, 0, 0])
    assert not env.get_state()["last_action_accepted"]
    np.testing.assert_array_equal(before, env.get_mask())
    viewer.reset_button.InvokeEvent("StateChangedEvent")
    np.testing.assert_array_equal(env.get_state()["phantom_T_probe_mm"], np.eye(4))
    assert all(abs(w.GetRepresentation().GetValue()) < 1e-8 for w in viewer.widgets.values())
    assert "frame" not in viewer.widgets
    viewer.export()
    assert list(Path("captures").glob("*.png"))
    viewer.reset()
    viewer.plotter.show(screenshot="desktop.png", auto_close=True)
    assert Path("desktop.png").stat().st_size > 10000
    result["gui_tested"] = True
    result["bundled_probe_model"] = True
result["audited_file_operations"] = audited_reads
Path("verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
'''
        environment = os.environ.copy()
        environment.update(PYTHONPATH=str(installed),
                           PHANTOM_DENY_ROOTS=json.dumps([
                               str(root), str(root.parent / "GPIS"),
                               "/home/wu/Downloads/linear_transducer.STL",
                           ]),
                           PHANTOM_VERIFY_GUI="1" if args.gui else "0",
                           MPLCONFIGDIR=str(temporary / "mpl-cache"))
        subprocess.run([sys.executable, "-c", code], cwd=temporary, env=environment, check=True)
        args.output.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(temporary / "verification.json", args.output / "verification.json")
        if args.gui:
            shutil.copyfile(temporary / "desktop.png", args.output / "desktop.png")
    print("Verification report:", args.output)


if __name__ == "__main__":
    main()
