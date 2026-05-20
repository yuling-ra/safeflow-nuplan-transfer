from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dmpc_fm_cbf.nuplan_sdk_adapter import (
    NuPlanSimulationConfig,
    format_env_exports,
    shell_command_string,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print a nuPlan official simulation command.")
    parser.add_argument("--root", required=True, help="Workspace root, e.g. /new_world/cockatiel/ra")
    parser.add_argument("--devkit-root", required=True, help="Path to nuplan-devkit checkout")
    parser.add_argument("--planner", required=True, choices=["official_idm", "official_pdm_closed"])
    parser.add_argument("--output-dir", default=None, help="Output directory for nuPlan simulation logs")
    parser.add_argument("--scenario-filter", default="mini")
    parser.add_argument("--simulation", default="closed_loop_nonreactive_agents")
    parser.add_argument("--python-bin", default="python")
    parser.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help="Extra Hydra override, can be repeated",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else root / "safeflow-nuplan-transfer" / "dmpc_fm_cbf" / "notebooks" / "cache" / "nuplan_sdk_runs" / args.planner
    )

    cfg = NuPlanSimulationConfig(
        devkit_root=Path(args.devkit_root),
        data_root=root / "data",
        map_root=root / "maps",
        output_dir=output_dir,
        planner_key=args.planner,
        scenario_filter=args.scenario_filter,
        simulation=args.simulation,
        extra_overrides=list(args.extra_override),
        python_bin=args.python_bin,
    )

    print("# Environment")
    print(format_env_exports(cfg))
    print()
    print("# Command")
    print(shell_command_string(cfg))


if __name__ == "__main__":
    main()
