#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "supplementary"
    / "machine_readable"
    / "external_comparator_metrics_31_rows.csv"
)
CONTRACT = ROOT / "configs" / "article_table_2_contract.json"
OUTPUT = ROOT / "results" / "table_2_absolute_rmse.csv"

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
    "delta_rmse_ci99_low",
    "delta_rmse_ci99_high",
    "significance_tier",
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
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract.get("schema_version") != 1 or contract.get("article_table") != "Table 2":
        raise ValueError("unsupported article Table 2 contract")
    panels = [record["id"] for record in contract["panels"]]
    panel_labels = {record["id"]: record["label"] for record in contract["panels"]}
    expected_n = {record["id"]: int(record["n"]) for record in contract["panels"]}
    model_specs = contract["models"]
    rows: list[dict[str, object]] = []

    dlg_spec = model_specs[0]
    if dlg_spec["model_id"] != "dlg_sol":
        raise ValueError("DLG-Sol must be the first model in the article contract")
    for panel_order, panel in enumerate(panels, start=1):
        n = int(consistent_reference(source, panel, "n"))
        if n != expected_n[panel]:
            raise ValueError(f"{panel}: article-contract n={expected_n[panel]}, source n={n}")
        absolute_rmse = consistent_reference(source, panel, "dlg_sol_rmse")
        if f"{absolute_rmse:.3f}" != dlg_spec["display_rmse"][panel]:
            raise ValueError(f"{panel}: DLG-Sol RMSE does not match the locked article display")
        rows.append(
            {
                "table_order": dlg_spec["table_order"],
                "model_id": dlg_spec["model_id"],
                "variant": dlg_spec["variant"],
                "model_label": dlg_spec["model_label"],
                "evidence_class": dlg_spec["evidence_class"],
                "panel_order": panel_order,
                "panel": panel,
                "evaluation": panel_labels[panel],
                "n": n,
                "available": True,
                "absolute_rmse": absolute_rmse,
                "delta_rmse_dlg_minus_comparator": np.nan,
                "delta_rmse_ci95_low": np.nan,
                "delta_rmse_ci95_high": np.nan,
                "delta_rmse_ci99_low": np.nan,
                "delta_rmse_ci99_high": np.nan,
                "significance_tier": "",
                "paired_inference_vs_dlg": "reference",
            }
        )

    for spec in model_specs[1:]:
        subset = source[
            source.model_id.eq(spec["source_model_id"])
            & source.variant.eq(spec["source_variant"])
        ].set_index("evaluation_id")
        for panel_order, panel in enumerate(panels, start=1):
            available = panel in subset.index
            record = subset.loc[panel] if available else None
            expected_display = spec["display_rmse"][panel]
            if available != (expected_display is not None):
                raise ValueError(
                    f"{spec['model_id']}:{spec['variant']}:{panel}: availability differs from article contract"
                )
            absolute_rmse = float(record["comparator_rmse"]) if available else np.nan
            if available and f"{absolute_rmse:.3f}" != expected_display:
                raise ValueError(
                    f"{spec['model_id']}:{spec['variant']}:{panel}: RMSE does not match locked article display"
                )
            rows.append(
                {
                    "table_order": spec["table_order"],
                    "model_id": spec["model_id"],
                    "variant": spec["variant"],
                    "model_label": spec["model_label"],
                    "evidence_class": spec["evidence_class"],
                    "panel_order": panel_order,
                    "panel": panel,
                    "evaluation": panel_labels[panel],
                    "n": int(record["n"]) if available else np.nan,
                    "available": available,
                    "absolute_rmse": absolute_rmse,
                    "delta_rmse_dlg_minus_comparator": (
                        float(record["delta_rmse_dlg_minus_comparator"]) if available else np.nan
                    ),
                    "delta_rmse_ci95_low": (
                        float(record["delta_rmse_ci95_low"]) if available else np.nan
                    ),
                    "delta_rmse_ci95_high": (
                        float(record["delta_rmse_ci95_high"]) if available else np.nan
                    ),
                    "delta_rmse_ci99_low": (
                        float(record["delta_rmse_ci99_low"]) if available else np.nan
                    ),
                    "delta_rmse_ci99_high": (
                        float(record["delta_rmse_ci99_high"]) if available else np.nan
                    ),
                    "significance_tier": (
                        str(record["significance_tier"])
                        if available and pd.notna(record["significance_tier"])
                        else ""
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
