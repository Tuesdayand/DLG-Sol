#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "supplementary"
    / "machine_readable"
    / "table_s06_external_comparator_metrics_and_paired_ci_31_rows.csv"
)
OUTPUT = ROOT / "results" / "table_2_absolute_rmse.csv"

PANELS = ["R01", "R02", "R03", "R04", "R05", "R09"]
PANEL_LABELS = {
    "R01": "AqSolDBc OOF",
    "R02": "ComPlat OOF",
    "R03": "TDC OOF",
    "R04": "TDC supplied test",
    "R05": "JCheM supplied test",
    "R09": "ComPlat supplied test",
}

MODEL_SPECS = [
    {
        "table_order": 2,
        "source_model_id": "pnnl_gnn",
        "source_variant": "none",
        "model_id": "pnnl_gnn",
        "variant": "none",
        "model_label": "PNNL GNN",
        "evidence_class": "independently retrained adaptation",
    },
    {
        "table_order": 3,
        "source_model_id": "ali_xgb125",
        "source_variant": "none",
        "model_id": "ali_xgb125",
        "variant": "none",
        "model_label": "Ali XGB-125D",
        "evidence_class": "split-adapted schema/settings; ComPlat same-split reproduction",
    },
    {
        "table_order": 4,
        "source_model_id": "bhattacharya_roy",
        "source_variant": "interaction",
        "model_id": "bhattacharya_roy",
        "variant": "interaction",
        "model_label": "Bhattacharya--Roy MLP--GNN",
        "evidence_class": "independently retrained adaptation",
    },
    {
        "table_order": 5,
        "source_model_id": "consensus_gnn_retrained",
        "source_variant": "retrained_adaptation",
        "model_id": "ulrich_consensus_adaptation",
        "variant": "retrained",
        "model_label": "Consensus GNN",
        "evidence_class": "independently retrained official-code adaptation",
    },
    {
        "table_order": 6,
        "source_model_id": "consensus_gnn_author",
        "source_variant": "released",
        "model_id": "ulrich_consensus",
        "variant": "released",
        "model_label": "Consensus GNN",
        "evidence_class": "authors' released predictions",
    },
]

COLUMNS = [
    "table_order",
    "model_id",
    "variant",
    "model_label",
    "evidence_class",
    "panel_order",
    "panel",
    "evaluation",
    "n",
    "available",
    "absolute_rmse",
    "delta_rmse_dlg_minus_comparator",
    "delta_rmse_ci95_low",
    "delta_rmse_ci95_high",
    "paired_inference_vs_dlg",
    "is_lowest_point_estimate",
]


def consistent_reference(source: pd.DataFrame, panel: str, column: str) -> float:
    values = source.loc[source.evaluation_id.eq(panel), column].dropna().unique()
    if len(values) != 1:
        raise ValueError(f"{panel}: expected one consistent {column}, found {values!r}")
    return float(values[0])


def main() -> None:
    source = pd.read_csv(SOURCE)
    rows: list[dict[str, object]] = []

    for panel_order, panel in enumerate(PANELS, start=1):
        rows.append(
            {
                "table_order": 1,
                "model_id": "dlg_sol",
                "variant": "none",
                "model_label": "DLG-Sol",
                "evidence_class": "this work",
                "panel_order": panel_order,
                "panel": panel,
                "evaluation": PANEL_LABELS[panel],
                "n": int(consistent_reference(source, panel, "n")),
                "available": True,
                "absolute_rmse": consistent_reference(source, panel, "dlg_sol_rmse"),
                "delta_rmse_dlg_minus_comparator": np.nan,
                "delta_rmse_ci95_low": np.nan,
                "delta_rmse_ci95_high": np.nan,
                "paired_inference_vs_dlg": "reference",
            }
        )

    for spec in MODEL_SPECS:
        subset = source[
            source.model_id.eq(spec["source_model_id"])
            & source.variant.eq(spec["source_variant"])
        ].set_index("evaluation_id")
        for panel_order, panel in enumerate(PANELS, start=1):
            available = panel in subset.index
            record = subset.loc[panel] if available else None
            rows.append(
                {
                    "table_order": spec["table_order"],
                    "model_id": spec["model_id"],
                    "variant": spec["variant"],
                    "model_label": spec["model_label"],
                    "evidence_class": spec["evidence_class"],
                    "panel_order": panel_order,
                    "panel": panel,
                    "evaluation": PANEL_LABELS[panel],
                    "n": int(record["n"]) if available else np.nan,
                    "available": available,
                    "absolute_rmse": float(record["comparator_rmse"]) if available else np.nan,
                    "delta_rmse_dlg_minus_comparator": (
                        float(record["delta_rmse_dlg_minus_comparator"]) if available else np.nan
                    ),
                    "delta_rmse_ci95_low": (
                        float(record["delta_rmse_ci95_low"]) if available else np.nan
                    ),
                    "delta_rmse_ci95_high": (
                        float(record["delta_rmse_ci95_high"]) if available else np.nan
                    ),
                    "paired_inference_vs_dlg": str(record["inference"]) if available else "not available",
                }
            )

    output = pd.DataFrame(rows)
    output["is_lowest_point_estimate"] = False
    for panel, group in output[output.available].groupby("panel"):
        minimum = group.absolute_rmse.min()
        output.loc[group.index, "is_lowest_point_estimate"] = np.isclose(
            group.absolute_rmse.to_numpy(float), minimum, rtol=0, atol=1e-12
        )
    output = output.sort_values(["table_order", "panel_order"])[COLUMNS]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT, index=False)
    print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
