#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from dlg_sol.workflow import load_materialized_branch_input, load_training_execution_recipes, load_training_role_contracts
from dlg_sol.workflow.descriptor_execution import run_descriptor_unit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--descriptor-cache", type=Path)
    parser.add_argument("--reuse-run", type=Path)
    parser.add_argument("--xgboost-device", default="cpu")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--role-contracts", type=Path, default=Path("configs/training_role_contracts.json"))
    parser.add_argument("--recipes", type=Path, default=Path("configs/training_execution_recipes.json"))
    args = parser.parse_args()
    roles = load_training_role_contracts(args.role_contracts)
    recipes = load_training_execution_recipes(args.recipes, roles)
    branch_input = load_materialized_branch_input(args.materialized, recipes)
    run_descriptor_unit(branch_input, args.output, args.descriptor_cache, args.xgboost_device, args.n_jobs, args.reuse_run)


if __name__ == "__main__":
    main()
