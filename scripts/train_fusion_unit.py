#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from dlg_sol.workflow import load_materialized_branch_input, load_training_execution_recipes, load_training_role_contracts
from dlg_sol.workflow.fusion_execution import run_fusion_unit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialized", type=Path, required=True)
    parser.add_argument("--language-run", type=Path, required=True)
    parser.add_argument("--geometry-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--reuse-run", type=Path)
    parser.add_argument("--role-contracts", type=Path, default=Path("configs/training_role_contracts.json"))
    parser.add_argument("--recipes", type=Path, default=Path("configs/training_execution_recipes.json"))
    parser.add_argument("--fusion-config", type=Path, default=Path("configs/fusion_ensemble.json"))
    parser.add_argument("--adapter-config", type=Path, default=Path("configs/dataset_adapters.json"))
    args = parser.parse_args()
    roles = load_training_role_contracts(args.role_contracts)
    recipes = load_training_execution_recipes(args.recipes, roles)
    branch_input = load_materialized_branch_input(args.materialized, recipes)
    run_fusion_unit(branch_input, args.output, args.language_run, args.geometry_run, args.fusion_config, args.adapter_config, args.device, args.reuse_run)


if __name__ == "__main__":
    main()
