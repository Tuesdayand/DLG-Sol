from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..evaluation import load_acquisition_registry
from .package import load_benchmark_input_package, load_input_contract_registry


ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m dlg_sol.data")
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--contracts", default=str(ROOT / "configs" / "benchmark_input_contracts.json"))
    parser.add_argument("--acquisition", default=str(ROOT / "configs" / "data_acquisition.json"))
    arguments = parser.parse_args()
    contracts = load_input_contract_registry(arguments.contracts)
    acquisition = load_acquisition_registry(arguments.acquisition)
    package = load_benchmark_input_package(arguments.package, arguments.evaluation_id, contracts, acquisition)
    print(json.dumps({
        "status": "PASS",
        "evaluation_id": package.evaluation_id,
        "dataset_id": package.dataset_id,
        "molecule_rows": package.molecule_rows,
        "partition_rows": package.partition_rows,
        "scored_rows": package.scored_rows,
        "redistributed": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
