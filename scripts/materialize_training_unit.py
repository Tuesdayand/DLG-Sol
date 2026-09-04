#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dlg_sol.data import load_benchmark_input_package, load_input_contract_registry
from dlg_sol.evaluation import load_acquisition_registry
from dlg_sol.workflow import load_training_role_contracts, materialize_training_unit, write_materialized_training_unit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--evaluation-id", required=True, choices=("R01", "R02", "R03", "R04", "R05", "R09"))
    parser.add_argument("--branch-id", required=True, choices=("descriptor", "language", "geometry", "fusion_head", "blend_coefficient"))
    parser.add_argument("--fold-index", required=True, type=int)
    parser.add_argument("--complat-train-source")
    parser.add_argument("--r03-dependency-package")
    parser.add_argument("--r02-dependency-package")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    inputs = load_input_contract_registry(ROOT / "configs" / "benchmark_input_contracts.json")
    acquisition = load_acquisition_registry(ROOT / "configs" / "data_acquisition.json")
    package = load_benchmark_input_package(arguments.package, arguments.evaluation_id, inputs, acquisition)
    dependencies = {}
    if arguments.r03_dependency_package:
        dependencies["R03"] = load_benchmark_input_package(arguments.r03_dependency_package, "R03", inputs, acquisition)
    if arguments.r02_dependency_package:
        dependencies["R02"] = load_benchmark_input_package(arguments.r02_dependency_package, "R02", inputs, acquisition)
    roles = load_training_role_contracts(ROOT / "configs" / "training_role_contracts.json")
    unit = materialize_training_unit(
        package,
        roles,
        arguments.evaluation_id,
        arguments.branch_id,
        arguments.fold_index,
        dependency_packages=dependencies,
        complat_train_source=arguments.complat_train_source,
    )
    result = write_materialized_training_unit(unit, arguments.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
