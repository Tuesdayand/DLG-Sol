#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dlg_sol.data.complat import prepare_complat_input_package


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test")
    parser.add_argument("--evaluation-id", required=True, choices=("R02", "R09"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--retrieval-date", required=True)
    arguments = parser.parse_args()
    result = prepare_complat_input_package(arguments.train, arguments.test, arguments.output, arguments.evaluation_id, arguments.retrieval_date, ROOT / "configs" / "complat_partition_recipe.json")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
