#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from dlg_sol.workflow import load_materialized_branch_input, load_training_execution_recipes, load_training_role_contracts
from dlg_sol.workflow.geometry_execution import run_geometry_unit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--reuse-run", type=Path)
    parser.add_argument("--settings-run", type=Path)
    parser.add_argument("--role-contracts", type=Path, default=Path("configs/training_role_contracts.json"))
    parser.add_argument("--recipes", type=Path, default=Path("configs/training_execution_recipes.json"))
    parser.add_argument("--geometry-config", type=Path, default=Path("configs/geometry_branch.json"))
    args = parser.parse_args()
    roles = load_training_role_contracts(args.role_contracts)
    recipes = load_training_execution_recipes(args.recipes, roles)
    branch_input = load_materialized_branch_input(args.materialized, recipes)
    run_geometry_unit(branch_input, args.output, args.geometry_config, args.device, args.seed_offset, args.reuse_run, args.settings_run)


if __name__ == "__main__":
    main()
