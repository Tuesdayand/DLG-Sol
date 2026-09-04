#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dlg_sol.data.jchem import prepare_jchem_input_package


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--retrieval-date", required=True)
    arguments = parser.parse_args()
    result = prepare_jchem_input_package(arguments.source, arguments.output, arguments.retrieval_date, ROOT / "configs" / "jchem_partition_recipe.json")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
