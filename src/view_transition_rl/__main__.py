"""Project command router; only implemented capabilities are exposed."""

import argparse
import sys


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="view-transition-rl", description="View Transition RL project tools")
    commands = parser.add_subparsers(dest="component", required=True)
    commands.add_parser("sim", help="Vessel simulation, desktop GUI and geometric evaluation", add_help=False)
    # Forward the remaining arguments verbatim, including --help and --config.
    if arguments and arguments[0] == "sim":
        from .simulation.__main__ import main as simulation_main
        return simulation_main(arguments[1:])
    parser.parse_args(arguments)


if __name__ == "__main__":
    main()
