import argparse
import json

from pathlib import Path

from .environment import PHANTOMS, PhantomEnv


def main(argv=None):
    parser = argparse.ArgumentParser(description="Standalone vessel phantom; lengths in mm, API angles in radians.")
    parser.add_argument("--config", help="JSON overrides for PhantomConfig")
    parser.add_argument("--phantom", choices=PHANTOMS, default="reference", help="Built-in phantom")
    commands = parser.add_subparsers(dest="command", required=True)
    gui = commands.add_parser("gui", help="Interactive desktop viewer")
    gui.add_argument("--screenshot", help="Render one frame offscreen to PNG, then exit")
    gui.add_argument("--export-dir", default="outputs/simulation/captures")
    sample = commands.add_parser("sample", help="Export a mask and pose without a display")
    sample.add_argument("--output", default="outputs/simulation/sample", help="Output prefix without extension")
    sample.add_argument("--frame", type=int, help="Constrained replay frame (0..189)")
    sample.add_argument("--pose-json", help="4x4 phantom_T_probe_mm matrix or exported state JSON")
    evaluation = commands.add_parser("evaluate", help="Compare all original scan frames")
    evaluation.add_argument("--output", default="outputs/simulation/evaluation")
    commands.add_parser("info", help="Print calibration and initial state")
    args = parser.parse_args(argv)
    try:
        overrides = json.loads(Path(args.config).read_text()) if args.config else None
        env = PhantomEnv(phantom=args.phantom, config_overrides=overrides)
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    if args.command == "gui":
        from .gui import run_gui
        run_gui(env, screenshot=args.screenshot, export_dir=args.export_dir)
    elif args.command == "sample":
        if args.frame is not None and args.pose_json:
            parser.error("--frame and --pose-json are mutually exclusive")
        if args.frame is not None:
            try:
                result = env.replay_frame(args.frame)
            except (IndexError, ValueError) as exc:
                parser.error(str(exc))
            if not result.accepted:
                parser.error(result.reason)
        if args.pose_json:
            pose = json.loads(Path(args.pose_json).read_text())
            if isinstance(pose, dict):
                if pose.get("phantom_id", env.phantom_id) != env.phantom_id:
                    parser.error("Pose phantom_id does not match the selected phantom.")
                pose = pose["phantom_T_probe_mm"]
            result = env.set_pose(pose)
            if not result.accepted:
                parser.error(result.reason)
        print("\n".join(str(p) for p in env.export(args.output)))
    elif args.command == "evaluate":
        from .evaluate import evaluate
        try:
            print(json.dumps(evaluate(args.output, env), indent=2))
        except ValueError as exc:
            parser.error(str(exc))
    else:
        print(json.dumps({"calibration": env.calibration, "state": env.get_state()}, indent=2))


if __name__ == "__main__":
    main()
