#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "release_manifest.json"
TABLE_SPEC = ROOT / "configs" / "machine_readable_tables.json"
TABLE_DIR = ROOT / "supplementary" / "machine_readable"
ARTICLE_TABLE_3_CONTRACT = ROOT / "configs" / "article_table_3_contract.json"
ARTICLE_TABLE_4_CONTRACT = ROOT / "configs" / "article_table_4_contract.json"
ENVIRONMENT_SPEC = ROOT / "environment" / "environments.json"
EVALUATION_REGISTRY = ROOT / "configs" / "evaluations.json"
CORE_PARITY = ROOT / "results" / "verified_manifests" / "core_module_parity.json"
DESCRIPTOR_PROTOCOLS = ROOT / "configs" / "descriptor_protocols.json"
DESCRIPTOR_PARITY = ROOT / "results" / "verified_manifests" / "descriptor_branch_parity.json"
DESCRIPTOR_CLEAN_SMOKE = ROOT / "results" / "verified_manifests" / "descriptor_clean_environment_smoke.json"
TRAINING_PARITY = ROOT / "results" / "verified_manifests" / "training_orchestration_parity.json"
TRAINING_CLEAN_SMOKE = ROOT / "results" / "verified_manifests" / "training_orchestration_clean_environment_smoke.json"
LANGUAGE_VARIANTS = ROOT / "configs" / "chemberta_variants.json"
LANGUAGE_PARITY = ROOT / "results" / "verified_manifests" / "language_branch_parity.json"
LANGUAGE_CLEAN_SMOKE = ROOT / "results" / "verified_manifests" / "language_clean_environment_smoke.json"
GEOMETRY_CONFIG = ROOT / "configs" / "geometry_branch.json"
GEOMETRY_PARITY = ROOT / "results" / "verified_manifests" / "geometry_branch_parity.json"
GEOMETRY_CLEAN_SMOKE = ROOT / "results" / "verified_manifests" / "geometry_clean_environment_smoke.json"
FUSION_CONFIG = ROOT / "configs" / "fusion_ensemble.json"
FUSION_PARITY = ROOT / "results" / "verified_manifests" / "fusion_branch_parity.json"
FUSION_CLEAN_SMOKE = ROOT / "results" / "verified_manifests" / "fusion_clean_environment_smoke.json"
ADAPTER_CONFIG = ROOT / "configs" / "dataset_adapters.json"
ADAPTER_PARITY = ROOT / "results" / "verified_manifests" / "dataset_adapter_parity.json"
ADAPTER_CLEAN_SMOKE = ROOT / "results" / "verified_manifests" / "dataset_adapter_clean_environment_smoke.json"
DATA_ACQUISITION = ROOT / "configs" / "data_acquisition.json"
BENCHMARK_INPUT_CONTRACTS = ROOT / "configs" / "benchmark_input_contracts.json"
AQSOLDBC_PARTITION_RECIPE = ROOT / "configs" / "aqsoldbc_partition_recipe.json"
AQSOLDBC_RECONSTRUCTION = ROOT / "results" / "verified_manifests" / "aqsoldbc_input_reconstruction.json"
COMPLAT_PARTITION_RECIPE = ROOT / "configs" / "complat_partition_recipe.json"
COMPLAT_RECONSTRUCTION = ROOT / "results" / "verified_manifests" / "complat_input_reconstruction.json"
JCHEM_PARTITION_RECIPE = ROOT / "configs" / "jchem_partition_recipe.json"
JCHEM_RECONSTRUCTION = ROOT / "results" / "verified_manifests" / "jchem_input_reconstruction.json"
TDC_PARTITION_RECIPE = ROOT / "configs" / "tdc_partition_recipe.json"
TDC_RECONSTRUCTION = ROOT / "results" / "verified_manifests" / "tdc_input_reconstruction.json"
TDC_INPUT_PREPARATION_CLEAN_SMOKE = ROOT / "results" / "verified_manifests" / "tdc_input_preparation_clean_environment_smoke.json"
INPUT_PREPARATION_CLEAN_SMOKE = ROOT / "results" / "verified_manifests" / "input_preparation_clean_environment_smoke.json"
EVALUATION_RUNNER_PARITY = ROOT / "results" / "verified_manifests" / "evaluation_runner_parity.json"
EVALUATION_RUNNER_CLEAN_SMOKE = ROOT / "results" / "verified_manifests" / "evaluation_runner_clean_environment_smoke.json"
TRAINING_ROLE_CONTRACTS = ROOT / "configs" / "training_role_contracts.json"
TRAINING_ROLE_PARITY = ROOT / "results" / "verified_manifests" / "training_role_contract_parity.json"
TRAINING_ROLE_MATERIALIZATION_PARITY = ROOT / "results" / "verified_manifests" / "training_role_materialization_parity.json"
TRAINING_EXECUTION_RECIPES = ROOT / "configs" / "training_execution_recipes.json"
TRAINING_EXECUTION_PARITY = ROOT / "results" / "verified_manifests" / "training_execution_recipe_parity.json"
BRANCH_EXECUTION_INTEGRATION = ROOT / "results" / "verified_manifests" / "branch_execution_integration.json"
EXTERNAL_SOURCES = ROOT / "configs" / "external_sources.json"
EXTERNAL_ADAPTER_PROVENANCE = ROOT / "configs" / "external_adapter_provenance.json"
REDISTRIBUTION_DECISIONS = ROOT / "configs" / "redistribution_decisions.json"
EXTERNAL_ADAPTER_PROVENANCE_AUDIT = ROOT / "results" / "verified_manifests" / "external_adapter_provenance_audit.json"
REDISTRIBUTION_DECISION_AUDIT = ROOT / "results" / "verified_manifests" / "redistribution_decision_audit.json"
V1C_RESULT_REGENERATION_AUDIT = ROOT / "results" / "verified_manifests" / "v1c_result_regeneration_audit.json"
TABLE_3_SOURCE_CONTEXT_AUDIT = ROOT / "results" / "verified_manifests" / "table_3_source_context_audit.json"
TABLE_3_SOURCE_CONTEXT = TABLE_DIR / "source_paper_benchmark_context.csv"
TABLE_4 = ROOT / "results" / "table_4_absolute_rmse.csv"
LICENSE_FILE = ROOT / "LICENSE"

sys.path.insert(0, str(ROOT / "src"))
from dlg_sol.provenance import validate_provenance_contract

TEXT_SUFFIXES = {".cff", ".csv", ".json", ".md", ".py", ".txt", ".yaml", ".yml"}
TEXT_DOTFILES = {".gitattributes", ".gitignore"}
INTERNAL_NAMES = ("code" + "x", "clau" + "de", "hand" + "off")
FORBIDDEN_PATTERNS = {
    "internal_agent_or_workflow_name": re.compile(
        r"\b(?:" + "|".join(INTERNAL_NAMES) + r")\b|"
        + "agent" + r"[_ -]?" + "work|"
        + "ai" + r"[- ]?(?:" + "generated|written|authored)|"
        + "written by " + "ai|generated by " + "ai",
        re.IGNORECASE,
    ),
    "private_absolute_path": re.compile(r"/(?:data|home)/koo/", re.IGNORECASE),
    "cluster_identifier": re.compile(r"(?:fai00[0-9]|goo[0-9]{4,})", re.IGNORECASE),
    "credential_assignment": re.compile(
        r"(?:api[_-]?key|access[_-]?token|secret|password)\s*[:=]\s*['\"][^'\"]+",
        re.IGNORECASE,
    ),
}

EXPECTED_EVALUATIONS = {
    "R01": ("out_of_fold", 8047, 5),
    "R02": ("out_of_fold", 17937, 5),
    "R03": ("out_of_fold", 7985, 5),
    "R04": ("supplied_test", 1997, None),
    "R05": ("supplied_test", 980, None),
    "R09": ("supplied_test", 1282, None),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(errors: list[str]) -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if payload.get("release_stage") != "V1_REPRODUCIBILITY_PACKAGE_RELEASE":
        errors.append("release stage does not match the selected v1.0 reproducibility scope")
    actual = {
        path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*")
        if path.is_file() and path != MANIFEST
        and not any(part in {".git", "__pycache__"} for part in path.relative_to(ROOT).parts)
    }
    if actual != set(payload["files"]):
        errors.append("release manifest file membership mismatch")
    for relative, expected in payload["files"].items():
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"missing manifest file: {relative}")
            continue
        if path.stat().st_size != expected["bytes"]:
            errors.append(f"byte-size mismatch: {relative}")
        if sha256(path) != expected["sha256"]:
            errors.append(f"sha256 mismatch: {relative}")


def validate_release_policy(errors: list[str]) -> None:
    text = LICENSE_FILE.read_text(encoding="utf-8") if LICENSE_FILE.is_file() else ""
    if not text.startswith("MIT License\n"):
        errors.append("MIT licence file is missing or malformed")
    if "Copyright (c) 2026 Hyunmo Goo and Hyeongwoo Kong" not in text:
        errors.append("MIT copyright attribution is missing")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    required_citation_lines = {
        "version: 1.0.4",
        "date-released: 2026-09-22",
        'title: "DLG-Sol combines molecular descriptors with language and three-dimensional embeddings for aqueous solubility prediction"',
        'repository-code: "https://github.com/Tuesdayand/DLG-Sol"',
    }
    if not required_citation_lines.issubset(set(citation.splitlines())):
        errors.append("v1.0 GitHub release citation metadata is incomplete")
    public_text = "\n".join(
        (ROOT / relative).read_text(encoding="utf-8")
        for relative in ("README.md", "data/README.md", "THIRD_PARTY_NOTICES.md", "docs/V1_RELEASE_PLAN.md")
    ).lower()
    obsolete_release_phrases = (
        "pre-release",
        "archival repository doi",
        "archival doi assigned",
        "does not require a separate archival doi",
        "github-only release policy",
    )
    if any(phrase in public_text for phrase in obsolete_release_phrases):
        errors.append("obsolete pre-release, GitHub-only, or archival-identifier policy remains")


def validate_tables(errors: list[str]) -> None:
    specs = json.loads(TABLE_SPEC.read_text(encoding="utf-8"))["tables"]
    actual_names = {path.name for path in TABLE_DIR.glob("*.csv")}
    if actual_names != set(specs):
        errors.append("machine-readable CSV set differs from the declared table specification")
    for name, expected in specs.items():
        path = TABLE_DIR / name
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration:
                errors.append(f"empty CSV: {name}")
                continue
            rows = sum(1 for _ in reader)
        if len(header) != expected["columns"]:
            errors.append(f"column-count mismatch: {name}")
        if rows != expected["rows"]:
            errors.append(f"row-count mismatch: {name}")

    required_columns = {
        "external_comparator_metrics_31_rows.csv": {
            "delta_rmse_ci95_low",
            "delta_rmse_ci95_high",
            "delta_rmse_ci99_low",
            "delta_rmse_ci99_high",
            "significance_tier",
            "main_table3_comparison",
        },
        "main_evaluation_effects_12_rows.csv": {
            "dlg_sol_mae",
            "component_mae",
            "delta_mae",
            "ci95_low",
            "ci95_high",
            "ci99_low",
            "ci99_high",
            "star",
        },
    }
    for name, expected_columns in required_columns.items():
        path = TABLE_DIR / name
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8-sig") as handle:
            header = set(next(csv.reader(handle), []))
        missing = expected_columns - header
        if missing:
            errors.append(f"confidence-interval or article-mapping columns missing from {name}: {sorted(missing)}")


def validate_weighting_summary(errors: list[str]) -> None:
    path = TABLE_DIR / "molecule_specific_weighting_14_rows.csv"
    if not path.is_file():
        errors.append("molecule-specific weighting summary is missing")
        return
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"panel", "n", "fixed_rmse", "policy", "gate_rmse", "delta_rmse",
                    "bootstrap_mean_delta_rmse", "ci95_low", "ci95_high", "n_boot", "seed"}
        if set(reader.fieldnames or []) != required:
            errors.append("weighting summary column schema mismatch")
            return
        rows = list(reader)
    panels = {"AqSolDBc_oof": 8047, "ComPlat_oof": 17937, "TDC_oof": 7985,
              "TDC_official": 1997, "JCheM_official": 980, "ComPlat_official": 1282,
              "Biogen": 2153}
    policies = {"cluster_soft", "pca_continuous"}
    observed = [(row["panel"], row["policy"]) for row in rows]
    if len(rows) != 14 or set(observed) != {(p, g) for p in panels for g in policies}:
        errors.append("weighting summary panel/policy membership mismatch")
        return
    with (TABLE_DIR / "main_evaluation_effects_12_rows.csv").open(encoding="utf-8", newline="") as handle:
        components = list(csv.DictReader(handle))
    panel_to_id = {"AqSolDBc_oof": "R01", "ComPlat_oof": "R02", "TDC_oof": "R03",
                   "TDC_official": "R04", "JCheM_official": "R05", "ComPlat_official": "R09"}
    for row in rows:
        try:
            values = {key: float(row[key]) for key in required - {"panel", "policy"}}
            if not all(math.isfinite(v) for v in values.values()):
                raise ValueError("nonfinite value")
            if int(row["n"]) != panels[row["panel"]] or int(row["n_boot"]) != 10000 or int(row["seed"]) <= 0:
                raise ValueError("sample count or bootstrap settings")
            if not math.isclose(values["gate_rmse"] - values["fixed_rmse"], values["delta_rmse"], rel_tol=0, abs_tol=1e-12):
                raise ValueError("gate-minus-fixed difference")
            if values["fixed_rmse"] < 0 or values["gate_rmse"] < 0 or values["ci95_low"] > values["ci95_high"]:
                raise ValueError("RMSE or interval bounds")
            if row["panel"] in panel_to_id:
                reference = [r for r in components if r["evaluation_id"] == panel_to_id[row["panel"]]]
                if len(reference) != 2 or any(not math.isclose(values["fixed_rmse"], float(r["dlg_sol_rmse"]), rel_tol=0, abs_tol=1e-12) for r in reference):
                    raise ValueError("fixed RMSE differs from component comparison")
        except (KeyError, ValueError, TypeError) as exc:
            errors.append(f"invalid weighting row {row['panel']}:{row['policy']}: {exc}")


def validate_article_table_3(errors: list[str]) -> None:
    contract = json.loads(ARTICLE_TABLE_3_CONTRACT.read_text(encoding="utf-8"))
    if contract.get("schema_version") != 1 or contract.get("article_table") != "Table 3":
        errors.append("unsupported Article Table 3 contract")
        return
    if contract.get("source") != TABLE_3_SOURCE_CONTEXT.relative_to(ROOT).as_posix():
        errors.append("Article Table 3 source path mismatch")
    with TABLE_3_SOURCE_CONTEXT.open(encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    by_id = {row.get("context_id"): row for row in source_rows}
    expected = contract.get("rows", [])
    expected_ids = [row.get("context_id") for row in expected]
    panel_ids = [context_id for panel in contract.get("panels", []) for context_id in panel.get("context_ids", [])]
    source_ids = [row.get("context_id") for row in source_rows]
    if len(source_rows) != 18 or len(by_id) != 18 or expected_ids != panel_ids or source_ids != expected_ids:
        errors.append("Article Table 3 source-context membership or order mismatch")
        return

    benchmark_by_id = {
        context_id: panel.get("benchmark")
        for panel in contract.get("panels", [])
        for context_id in panel.get("context_ids", [])
    }

    def displayed(value: str, template: str) -> str:
        plain = template.strip("()")
        decimals = len(plain.partition(".")[2])
        rendered = f"{float(value):.{decimals}f}"
        return f"({rendered})" if template.startswith("(") else rendered

    for spec in expected:
        context_id = spec.get("context_id")
        row = by_id.get(context_id)
        if row is None:
            errors.append(f"Article Table 3 source-context row missing: {context_id}")
            continue
        if row.get("benchmark") != benchmark_by_id.get(context_id):
            errors.append(f"Article Table 3 benchmark mismatch: {context_id}")
        if row.get("model_or_method") != spec.get("model_or_method") or int(row.get("n", 0)) != spec.get("n"):
            errors.append(f"Article Table 3 model or n mismatch: {context_id}")
        if displayed(row.get("rmse", ""), spec.get("main_rmse", "")) != spec.get("main_rmse"):
            errors.append(f"Article Table 3 main-text RMSE mismatch: {context_id}")
        if displayed(row.get("rmse", ""), spec.get("supplementary_rmse", "")) != spec.get("supplementary_rmse"):
            errors.append(f"Article Table 3 Supplementary RMSE mismatch: {context_id}")
        expected_mae = spec.get("supplementary_mae")
        if expected_mae is None:
            if row.get("mae") or not row.get("mae_status", "").startswith("not_reported"):
                errors.append(f"Article Table 3 MAE availability mismatch: {context_id}")
        elif displayed(row.get("mae", ""), expected_mae) != expected_mae:
            errors.append(f"Article Table 3 Supplementary MAE mismatch: {context_id}")
    for context_id, expected_scope in contract.get("required_qualifiers", {}).items():
        if by_id.get(context_id, {}).get("comparison_scope") != expected_scope:
            errors.append(f"Article Table 3 evidence qualifier mismatch: {context_id}")
    audit = json.loads(TABLE_3_SOURCE_CONTEXT_AUDIT.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        errors.append("Article Table 3 source-context audit status mismatch")
    audit_source = audit.get("source", {})
    audit_contract = audit.get("article_contract", {})
    if audit_source.get("path") != TABLE_3_SOURCE_CONTEXT.relative_to(ROOT).as_posix() or audit_source.get("sha256") != sha256(TABLE_3_SOURCE_CONTEXT):
        errors.append("Article Table 3 source-context audit source identity mismatch")
    if audit_contract.get("path") != ARTICLE_TABLE_3_CONTRACT.relative_to(ROOT).as_posix() or audit_contract.get("sha256") != sha256(ARTICLE_TABLE_3_CONTRACT):
        errors.append("Article Table 3 source-context audit contract identity mismatch")


def validate_article_table_4(errors: list[str]) -> None:
    contract = json.loads(ARTICLE_TABLE_4_CONTRACT.read_text(encoding="utf-8"))
    if contract.get("schema_version") != 1 or contract.get("article_table") != "Table 4":
        errors.append("unsupported Article Table 4 contract")
        return
    panels = contract.get("panels", [])
    models = contract.get("models", [])
    panel_ids = [record.get("id") for record in panels]
    if panel_ids != ["R01", "R02", "R03", "R04", "R05", "R09"]:
        errors.append("Article Table 4 panel order mismatch")
    if len(models) != 5 or models[0].get("model_id") != "dlg_sol":
        errors.append("Article Table 4 model contract mismatch")
        return

    with TABLE_4.open(encoding="utf-8", newline="") as handle:
        output_rows = list(csv.DictReader(handle))
    by_key = {(row.get("model_id"), row.get("variant"), row.get("panel")): row for row in output_rows}
    if len(output_rows) != 30 or len(by_key) != 30:
        errors.append("Article Table 4 output row identity mismatch")
        return

    expected_primary_source_rows: set[tuple[str, str, str]] = set()
    for model in models:
        for panel in panel_ids:
            key = (model.get("model_id"), model.get("variant"), panel)
            row = by_key.get(key)
            if row is None:
                errors.append(f"Article Table 4 row missing: {key}")
                continue
            for field in ("model_label", "evidence_class"):
                if row.get(field) != model.get(field):
                    errors.append(f"Article Table 4 {field} mismatch: {key}")
            expected_display = model.get("display_rmse", {}).get(panel)
            available = row.get("available") == "True"
            if available != (expected_display is not None):
                errors.append(f"Article Table 4 availability mismatch: {key}")
            if available and f"{float(row['absolute_rmse']):.3f}" != expected_display:
                errors.append(f"Article Table 4 displayed RMSE mismatch: {key}")
            if model.get("model_id") != "dlg_sol" and available:
                expected_primary_source_rows.add((model.get("source_model_id"), model.get("source_variant"), panel))
                if not row.get("delta_rmse_ci99_low") or not row.get("delta_rmse_ci99_high"):
                    errors.append(f"Article Table 4 99% CI missing: {key}")

    source_path = TABLE_DIR / "external_comparator_metrics_31_rows.csv"
    with source_path.open(encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    marked_main_text = {
        (row.get("model_id"), row.get("variant"), row.get("evaluation_id"))
        for row in source_rows
        if row.get("main_table3_comparison") == "True"
    }
    legacy_released_prediction = {("consensus_gnn_author", "released", "R05")}
    if marked_main_text != expected_primary_source_rows | legacy_released_prediction:
        errors.append("legacy machine-readable main-text membership differs from Article Tables 3--4")
    if any(
        row.get("model_id") == "bhattacharya_roy"
        and row.get("variant") == "interaction"
        and row.get("main_table3_comparison") == "True"
        for row in source_rows
    ):
        errors.append("Bhattacharya--Roy interaction sensitivity is incorrectly marked as a primary comparator")


def validate_environment_spec(errors: list[str]) -> None:
    payload = json.loads(ENVIRONMENT_SPEC.read_text(encoding="utf-8"))
    policy = payload.get("policy", {})
    if payload.get("schema_version") != 1 or policy.get("isolated_environments_required") is not True or policy.get("install_all_requirement_files_together") is not False:
        errors.append("environment-isolation policy is incomplete")
    records = payload.get("environments", [])
    required_ids = {
        "integrity",
        "analysis",
        "descriptor",
        "input_preparation",
        "tdc_input_preparation",
        "training_role_materialization",
        "language",
        "geometry",
        "fusion",
        "release_test",
    }
    if {record.get("id") for record in records} != required_ids:
        errors.append("machine-readable environment coverage mismatch")
    for record in records:
        requirement = record.get("requirements")
        if requirement and not (ROOT / "environment" / requirement).is_file():
            errors.append(f"environment requirement file missing: {requirement}")
        if not record.get("python_constraint") and not record.get("recommended_python"):
            errors.append(f"Python version declaration missing: {record.get('id')}")


def validate_evaluation_registry(errors: list[str]) -> None:
    payload = json.loads(EVALUATION_REGISTRY.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        errors.append("unsupported evaluation-registry schema")
    firewall = payload.get("common_label_firewall", {})
    if firewall.get("scored_labels_allowed_for") != ["scoring"]:
        errors.append("evaluation registry weakens the scored-label firewall")
    if firewall.get("scored_labels_forbidden_for") != ["preprocessing", "model_fitting", "coefficient_fitting"]:
        errors.append("evaluation registry misstates the common scored-label exclusions")
    if firewall.get("selection_policy") != "evaluation_specific":
        errors.append("evaluation registry does not expose evaluation-specific selection provenance")
    records = payload.get("evaluations", [])
    observed = {record.get("evaluation_id"): record for record in records}
    if set(observed) != set(EXPECTED_EVALUATIONS):
        errors.append("evaluation registry does not contain the exact primary evaluation set")
        return
    for evaluation_id, expected in EXPECTED_EVALUATIONS.items():
        record = observed[evaluation_id]
        actual = (record.get("design"), record.get("expected_rows"), record.get("outer_folds"))
        if actual != expected:
            errors.append(f"evaluation-registry mismatch: {evaluation_id}")
    expected_aggregation = {
        "R01": "one held-out prediction per outer fold",
        "R02": "one held-out prediction per outer fold",
        "R03": "one held-out prediction per outer fold",
        "R04": "mean of five split-model predictions per supplied-test row",
        "R05": "mean of five split-model predictions per supplied-test row",
        "R09": "one prediction per supplied-test row",
    }
    for evaluation_id, aggregation in expected_aggregation.items():
        if observed[evaluation_id].get("prediction_aggregation") != aggregation:
            errors.append(f"evaluation aggregation mismatch: {evaluation_id}")
    r02 = observed["R02"]
    if r02.get("display_name") != "ComPlat OOF after global preselection":
        errors.append("R02 display name incorrectly implies nested selection")
    if "not nested" not in r02.get("selection_label_access", ""):
        errors.append("R02 global preselection exception is not disclosed")


def validate_core_parity(errors: list[str]) -> None:
    payload = json.loads(CORE_PARITY.read_text(encoding="utf-8"))
    records = payload.get("evaluations", [])
    if payload.get("status") != "PASS":
        errors.append("canonical core parity status is not PASS")
    if {record.get("evaluation_id") for record in records} != set(EXPECTED_EVALUATIONS):
        errors.append("canonical core parity does not cover all primary evaluations")
    if sum(record.get("rows", 0) for record in records) != sum(item[1] for item in EXPECTED_EVALUATIONS.values()):
        errors.append("canonical core parity row total mismatch")
    if payload.get("max_metric_absolute_difference") != 0.0:
        errors.append("canonical metric parity mismatch")
    if payload.get("max_bootstrap_absolute_difference") != 0.0:
        errors.append("canonical paired-bootstrap parity mismatch")
    if payload.get("max_blend_reconstruction_absolute_difference", float("inf")) > 2e-7:
        errors.append("canonical fixed-blend reconstruction exceeds tolerance")


def validate_descriptor_branch(errors: list[str]) -> None:
    protocols = json.loads(DESCRIPTOR_PROTOCOLS.read_text(encoding="utf-8"))
    if protocols.get("schema_version") != 1:
        errors.append("unsupported descriptor-protocol schema")
        return
    records = protocols.get("protocols", {})
    expected_counts = {
        "aqsoldbc": [373, 353, 354, 352, 356],
        "tdc": [337, 341, 337, 340, 338],
        "complat": [646, 650, 654, 650, 650],
        "jchem": [571, 575, 572, 574, 615],
    }
    if set(records) != set(expected_counts):
        errors.append("descriptor protocol set is incomplete")
        return
    for name, expected in expected_counts.items():
        key = "expected_split_feature_counts" if name == "jchem" else "expected_outer_feature_counts"
        if records[name].get(key) != expected:
            errors.append(f"descriptor feature-count contract mismatch: {name}")
    rules = {
        "aqsoldbc": (0, 0.95, "gt"),
        "tdc": (1, 0.98, "ge"),
        "complat": (1, None, None),
        "jchem": (1, 0.98, "ge"),
    }
    for name, expected in rules.items():
        filtering = records[name]["filtering"]
        actual = (
            filtering.get("variance_ddof"),
            filtering.get("correlation_threshold"),
            filtering.get("correlation_comparison"),
        )
        if actual != expected:
            errors.append(f"descriptor filtering contract mismatch: {name}")
    parity = json.loads(DESCRIPTOR_PARITY.read_text(encoding="utf-8"))
    if parity.get("status") != "PASS":
        errors.append("descriptor branch parity status is not PASS")
    if parity.get("row_level_inputs_redistributed") is not False:
        errors.append("descriptor parity manifest misstates row-level redistribution")
    if set(parity.get("protocols", {})) != set(expected_counts):
        errors.append("descriptor parity does not cover every protocol")
        return
    for name, expected in expected_counts.items():
        record = parity["protocols"][name]
        if record.get("retained_feature_counts") != expected:
            errors.append(f"descriptor parity feature-count mismatch: {name}")
        if record.get("feature_schema_parity") is not True or record.get("candidate_parity") is not True:
            errors.append(f"descriptor parity check failed: {name}")
    if parity.get("synthetic_xgboost_constructor_max_absolute_prediction_difference") != 0.0:
        errors.append("descriptor XGBoost constructor parity mismatch")
    smoke = json.loads(DESCRIPTOR_CLEAN_SMOKE.read_text(encoding="utf-8"))
    if smoke.get("status") != "PASS" or smoke.get("tests_discovered") != 16:
        errors.append("descriptor clean-environment smoke test failed")
    if smoke.get("failures") != 0 or smoke.get("errors") != 0:
        errors.append("descriptor clean-environment tests were not clean")
    if smoke.get("mordred_descriptor_columns") != 1613:
        errors.append("descriptor clean-environment Mordred contract mismatch")


def validate_training_orchestration(errors: list[str]) -> None:
    parity = json.loads(TRAINING_PARITY.read_text(encoding="utf-8"))
    expected = {
        "aqsoldbc_selected_trials": [4, 8, 4, 0, 0],
        "aqsoldbc_final_tree_counts": [3456, 3332, 3064, 2601, 3198],
        "tdc_selected_trials": [7, 19, 5, 0, 7],
        "tdc_best_iterations": [230, 377, 650, 365, 955],
        "complat_selected_trial": 2,
        "jchem_selected_trials": [3, 15, 6, 6, 11],
    }
    if parity.get("status") != "PASS" or parity.get("gate") != "training_orchestration_o1":
        errors.append("training orchestration parity status is not PASS")
    if parity.get("row_level_inputs_redistributed") is not False:
        errors.append("training orchestration manifest misstates row-level redistribution")
    if parity.get("observed") != expected or parity.get("expected") != expected:
        errors.append("training orchestration historical selection parity mismatch")
    smoke = json.loads(TRAINING_CLEAN_SMOKE.read_text(encoding="utf-8"))
    if smoke.get("status") != "PASS" or smoke.get("tests_discovered") != 28:
        errors.append("training orchestration clean-environment smoke test failed")
    if smoke.get("failures") != 0 or smoke.get("errors") != 0:
        errors.append("training orchestration clean-environment tests were not clean")


def validate_language_branch(errors: list[str]) -> None:
    variants = json.loads(LANGUAGE_VARIANTS.read_text(encoding="utf-8"))
    if variants.get("schema_version") != 2:
        errors.append("unsupported ChemBERTa variant schema")
        return
    expected_scope = {
        "randomized_single_task": "final_dlg_sol_language_encoder",
        "canonical_multitask": "development_candidate_not_final_dlg_sol",
        "canonical_single_task": "development_candidate_not_final_dlg_sol",
    }
    if variants.get("final_model_profile") != "randomized_single_task" or variants.get("variant_scope") != expected_scope:
        errors.append("ChemBERTa final and development scopes are misstated")
    model = variants.get("pretrained_model", {})
    if model.get("identifier") != "seyonec/ChemBERTa-zinc-base-v1":
        errors.append("ChemBERTa pretrained identifier mismatch")
    if model.get("revision") is not None or model.get("revision_status") != "not_recorded_in_historical_run":
        errors.append("ChemBERTa revision provenance is misstated")
    records = variants.get("variants", {})
    expected = {
        "canonical_multitask": (5e-5, 0, "logS_plus_auxiliary", 0.15),
        "randomized_single_task": (2e-5, 2, "logS", 0.0),
        "canonical_single_task": (5e-5, 0, "logS", 0.0),
    }
    if set(records) != set(expected):
        errors.append("ChemBERTa variant set is incomplete")
        return
    for name, values in expected.items():
        record = records[name]
        observed = (
            record.get("learning_rate"),
            record.get("training_randomized_smiles_per_molecule"),
            record.get("task"),
            record.get("auxiliary_weight"),
        )
        if observed != values:
            errors.append(f"ChemBERTa variant contract mismatch: {name}")
    if records["canonical_multitask"].get("auxiliary_targets") != ["logp", "tpsa", "molwt", "hbd", "hba"]:
        errors.append("ChemBERTa auxiliary-target contract mismatch")
    parity = json.loads(LANGUAGE_PARITY.read_text(encoding="utf-8"))
    if parity.get("status") != "PASS" or parity.get("gate") != "chemberta_language_l1":
        errors.append("ChemBERTa language parity status is not PASS")
    if parity.get("row_level_inputs_redistributed") is not False or parity.get("checkpoints_redistributed") is not False or parity.get("embeddings_redistributed") is not False:
        errors.append("ChemBERTa parity manifest misstates redistribution")
    if parity.get("pretrained_revision_recorded") is not False or parity.get("configuration_parity") is not True:
        errors.append("ChemBERTa configuration provenance mismatch")
    auxiliary = parity.get("auxiliary_target_parity", {})
    if auxiliary.get("rows") != 10298 or auxiliary.get("columns") != ["logp", "tpsa", "molwt", "hbd", "hba"]:
        errors.append("ChemBERTa auxiliary-target parity shape mismatch")
    if parity.get("auxiliary_absolute_tolerance") != 1e-12 or parity.get("embedding_absolute_tolerance") != 3e-6:
        errors.append("ChemBERTa parity tolerance contract mismatch")
    if auxiliary.get("archived_missing_values") != 0 or auxiliary.get("recomputed_missing_values") != 0 or auxiliary.get("maximum_absolute_difference", float("inf")) > 1e-12:
        errors.append("ChemBERTa auxiliary-target parity mismatch")
    embeddings = parity.get("embedding_extraction_parity", [])
    if {record.get("variant") for record in embeddings} != set(expected):
        errors.append("ChemBERTa embedding parity does not cover every variant")
    for record in embeddings:
        if record.get("archived_embedding_shape") != [10298, 768] or record.get("observed_embedding_shape") != [16, 768]:
            errors.append("ChemBERTa embedding shape mismatch")
        if record.get("maximum_absolute_difference", float("inf")) > 3e-6 or record.get("finite") is not True:
            errors.append("ChemBERTa embedding extraction parity mismatch")
    smoke = json.loads(LANGUAGE_CLEAN_SMOKE.read_text(encoding="utf-8"))
    if smoke.get("status") != "PASS" or smoke.get("tests_discovered") != 37:
        errors.append("ChemBERTa clean-environment smoke test failed")
    if smoke.get("failures") != 0 or smoke.get("errors") != 0 or smoke.get("skipped") != 0:
        errors.append("ChemBERTa clean-environment tests were not clean")


def validate_geometry_branch(errors: list[str]) -> None:
    config = json.loads(GEOMETRY_CONFIG.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        errors.append("unsupported geometry configuration schema")
        return
    graph = config.get("graph_construction", {})
    model = config.get("model", {})
    dimensions = (graph.get("node_dimension"), graph.get("edge_dimension"), graph.get("auxiliary_dimension"))
    if dimensions != (42, 24, 19):
        errors.append("geometry feature-dimension contract mismatch")
    graph_contract = (
        graph.get("conformers_attempted"),
        graph.get("conformers_retained"),
        graph.get("proximity_cutoff_angstrom"),
        graph.get("explicit_hydrogen_policy"),
    )
    if graph_contract != (5, 3, 4.5, "hydrogens_bonded_to_nitrogen_or_oxygen"):
        errors.append("geometry graph-construction contract mismatch")
    if model.get("architecture") != "distance_aware_scalar_3d_mpnn" or model.get("coordinate_updates") is not False:
        errors.append("geometry model architecture is misstated")
    if model.get("checkpoint_selection") != "selection_rmse":
        errors.append("geometry checkpoint-selection contract mismatch")
    parity = json.loads(GEOMETRY_PARITY.read_text(encoding="utf-8"))
    if parity.get("status") != "PASS" or parity.get("gate") != "geometry_branch_g1":
        errors.append("geometry branch parity status is not PASS")
    redistribution = (
        parity.get("row_level_inputs_redistributed"),
        parity.get("conformer_caches_redistributed"),
        parity.get("coordinates_redistributed"),
        parity.get("checkpoints_redistributed"),
        parity.get("embeddings_redistributed"),
    )
    if redistribution != (False, False, False, False, False):
        errors.append("geometry parity manifest misstates redistribution")
    cache = parity.get("development_cache", {})
    if (cache.get("records"), cache.get("graphs")) != (10298, 24457):
        errors.append("geometry development-cache shape mismatch")
    probe = parity.get("graph_reconstruction_probe", {})
    if (probe.get("rows"), probe.get("graphs")) != (16, 36):
        errors.append("geometry graph-reconstruction probe shape mismatch")
    if probe.get("record_graph_counts_equal") is not True or probe.get("edge_indices_equal") is not True:
        errors.append("geometry graph-reconstruction topology mismatch")
    if any(value != 0.0 for value in probe.get("maximum_absolute_difference", {}).values()):
        errors.append("geometry graph-reconstruction value mismatch")
    checkpoints = parity.get("checkpoint_strict_loads", [])
    if len(checkpoints) != 5 or not all(record.get("strict_load") is True for record in checkpoints):
        errors.append("geometry archived-checkpoint strict-load coverage mismatch")
    embedding = parity.get("embedding_extraction_probe", {})
    if embedding.get("archived_embedding_shape") != [10298, 256] or embedding.get("rows") != 16:
        errors.append("geometry embedding parity shape mismatch")
    if embedding.get("absolute_tolerance") != 3e-6 or embedding.get("maximum_absolute_difference", float("inf")) > 3e-6:
        errors.append("geometry embedding parity mismatch")
    if embedding.get("embedding_plus_auxiliary_maximum_absolute_difference", float("inf")) > 3e-6 or embedding.get("finite") is not True:
        errors.append("geometry embedding-plus-auxiliary parity mismatch")
    prediction = parity.get("prediction_probe", {})
    if prediction.get("rows") != 16 or prediction.get("absolute_tolerance") != 3e-6 or prediction.get("maximum_absolute_difference", float("inf")) > 3e-6 or prediction.get("finite") is not True:
        errors.append("geometry prediction parity mismatch")
    if parity.get("coordinate_updates") is not False or parity.get("scored_inference_accepts_labels") is not False:
        errors.append("geometry architecture or label-firewall manifest mismatch")
    smoke = json.loads(GEOMETRY_CLEAN_SMOKE.read_text(encoding="utf-8"))
    if smoke.get("status") != "PASS" or smoke.get("tests_discovered") != 44 or smoke.get("geometry_tests") != 7:
        errors.append("geometry clean-environment smoke test failed")
    if smoke.get("failures") != 0 or smoke.get("errors") != 0 or smoke.get("skipped") != 0:
        errors.append("geometry clean-environment tests were not clean")
    if smoke.get("validator_counterexamples") != 2 or smoke.get("coordinate_update_misstatement_rejected") is not True or smoke.get("embedding_parity_mismatch_rejected") is not True:
        errors.append("geometry validator counterexamples were not recorded")


def validate_fusion_branch(errors: list[str]) -> None:
    config = json.loads(FUSION_CONFIG.read_text(encoding="utf-8"))
    if config.get("schema_version") != 2 or config.get("model_identity") != "aug2_head4":
        errors.append("unsupported fusion configuration schema")
        return
    expected = [
        (0, "randomized_single_task", "raw_standardized_concat", "residual_mlp", 260901, 29),
        (1, "randomized_single_task", "raw_standardized_concat", "residual_mlp", 260902, 82),
        (2, "randomized_single_task", "raw_standardized_concat", "residual_mlp", 260903, 40),
        (3, "randomized_single_task", "raw_standardized_concat", "residual_mlp", 260904, 21),
    ]
    observed = [
        (
            record.get("member_index"),
            record.get("language_variant"),
            record.get("preprocessing"),
            record.get("regressor"),
            record.get("base_seed"),
            record.get("fixed_epochs"),
        )
        for record in config.get("members", [])
    ]
    if observed != expected:
        errors.append("fusion-head mapping differs from the canonical aug2_head4 contract")
    preprocessing = config.get("preprocessing", {})
    if preprocessing.get("fit_partition_only") is not True or preprocessing.get("method") != "modality_wise_standardization_then_raw_concatenation":
        errors.append("fusion preprocessing contract mismatch")
    if preprocessing.get("pca_used_by_final_model") is not False or preprocessing.get("l2_normalization_used_by_final_model") is not False:
        errors.append("PCA or L2 normalization is incorrectly represented as final")
    if preprocessing.get("adapter_winsorization") != {"aqsoldbc": True, "complat": True, "tdc": True, "jchem": False}:
        errors.append("fusion adapter winsorization mapping mismatch")
    if preprocessing.get("failed_geometry_fallback_uses_labels") is not False:
        errors.append("fusion adapter or fallback boundary is misstated")
    architecture = config.get("head_architecture", {})
    if architecture != {"hidden": 384, "depth": 2, "dropout": 0.45, "output_dimension": 1}:
        errors.append("fusion residual-head architecture mismatch")
    training = config.get("training", {})
    expected_training = {
        "loss": "mean_squared_error",
        "optimizer": "AdamW",
        "learning_rate": 0.0005,
        "weight_decay": 0.0005,
        "batch_size": 128,
        "gradient_clip_norm": 5.0,
        "epoch_policy": "fixed_per_head",
        "fold_seed_rule": "base_seed_plus_100000_times_fold_index",
        "schedule_provenance": "per-seed median best epoch from AqSolDBc five-fold development",
    }
    if training != expected_training:
        errors.append("fusion fixed-head training policy mismatch")
    prediction = config.get("neural_prediction", {})
    if (
        prediction.get("aggregation"),
        prediction.get("expected_members"),
        prediction.get("shared_input_representation"),
        prediction.get("input_dependent_weights"),
    ) != ("arithmetic_mean", 4, True, False):
        errors.append("neural-member aggregation contract mismatch")
    parity = json.loads(FUSION_PARITY.read_text(encoding="utf-8"))
    if parity.get("status") != "PASS" or parity.get("gate") != "canonical_aug2_head4_identity_f2" or parity.get("model_identity") != "aug2_head4":
        errors.append("fusion branch parity status is not PASS")
    redistribution = (
        parity.get("row_level_inputs_redistributed"),
        parity.get("checkpoints_redistributed"),
        parity.get("embeddings_redistributed"),
        parity.get("predictions_redistributed"),
    )
    if redistribution != (False, False, False, False):
        errors.append("fusion parity manifest misstates redistribution")
    parity_mapping = [
        (
            record.get("member_index"),
            record.get("language_variant"),
            record.get("preprocessing"),
            record.get("regressor"),
            record.get("base_seed"),
            record.get("fixed_epochs"),
        )
        for record in parity.get("members", [])
    ]
    if parity_mapping != expected:
        errors.append("fusion parity head mapping mismatch")
    evaluations = {record.get("evaluation_id"): record for record in parity.get("evaluations", [])}
    expected_rows = {"R01": 8047, "R02": 17937, "R03": 7985, "R04": 1997, "R05": 980, "R09": 1282}
    if set(evaluations) != set(expected_rows):
        errors.append("fusion identity parity evaluation set mismatch")
    else:
        for evaluation_id, rows in expected_rows.items():
            record = evaluations[evaluation_id]
            if record.get("rows") != rows or record.get("record_id_unique_in_both_sources") is not True or record.get("record_id_coverage_complete") is not True:
                errors.append(f"fusion row identity coverage mismatch: {evaluation_id}")
            if record.get("maximum_absolute_prediction_difference") != 0.0:
                errors.append(f"fusion neural prediction identity mismatch: {evaluation_id}")
    if parity.get("all_six_evaluations_match") is not True or parity.get("maximum_absolute_prediction_difference") != 0.0:
        errors.append("fusion six-evaluation identity summary mismatch")
    if parity.get("scored_inference_accepts_labels") is not False:
        errors.append("fusion scored-inference label firewall is misstated")
    smoke = json.loads(FUSION_CLEAN_SMOKE.read_text(encoding="utf-8"))
    if smoke.get("status") != "PASS" or smoke.get("tests_discovered") < 55 or smoke.get("fusion_tests") < 11:
        errors.append("fusion clean-environment smoke test failed")
    if smoke.get("failures") != 0 or smoke.get("errors") != 0 or smoke.get("skipped") != 0:
        errors.append("fusion clean-environment tests were not clean")
    if smoke.get("validator_counterexamples", 0) < 3 or smoke.get("universal_winsorization_misstatement_rejected") is not True or smoke.get("incomplete_member_mapping_rejected") is not True or smoke.get("prediction_identity_mismatch_rejected") is not True:
        errors.append("fusion validator counterexamples were not recorded")


def validate_dataset_adapters(errors: list[str]) -> None:
    config = json.loads(ADAPTER_CONFIG.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("model_identity") != "aug2_head4" or config.get("fold_seed_multiplier") != 100000:
        errors.append("unsupported dataset-adapter configuration")
        return
    output = config.get("common_output_schema", {})
    if output != {"record_key": "record_id", "fold_column": "fold_index", "member_columns": ["head_0", "head_1", "head_2", "head_3"], "canonical_neural_column": "aug2_head4"}:
        errors.append("dataset-adapter output schema mismatch")
    records = {record.get("evaluation_id"): record for record in config.get("evaluation_adapters", [])}
    expected = {
        "R01": ("aqsoldbc", "split", ["Train"], [], ["External"], ["Val"], [0, 1, 2, 3, 4], True, 0.001, "one_held_out_prediction", 8047, "prespecified_0.5", "aug2_head4"),
        "R02": ("complat", "split", ["Train"], [], ["External"], [], [0, 1, 2, 3, 4], True, 0.001, "one_held_out_prediction", 17937, "cross_fitted_from_training_side_oof", "aug2_head4"),
        "R03": ("tdc", "partition", ["fit"], [], ["oof_holdout"], ["selection", "official_test"], [0, 1, 2, 3, 4], True, 0.001, "one_held_out_prediction", 7985, "cross_fitted_from_training_side_oof", "aug2_head4"),
        "R04": ("tdc", "partition", ["fit"], [], ["official_test"], ["selection", "oof_holdout"], [0, 1, 2, 3, 4], True, 0.001, "mean_across_split_models", 1997, "one_coefficient_from_complete_tdc_oof", "aug2_head4"),
        "R05": ("jchem", "split", ["Train"], ["Val"], ["External"], [], [1, 2, 3, 4, 5], False, None, "mean_across_split_models", 980, "validation_only_within_each_reported_split", "aug2_head4"),
        "R09": ("complat", "split", ["Train"], [], ["External"], [], [0], True, 0.001, "single_full_refit", 1282, "one_coefficient_from_complete_complat_oof", "polarh_aug2_head4"),
    }
    if set(records) != set(expected):
        errors.append("dataset-adapter evaluation set mismatch")
    else:
        for evaluation_id, contract in expected.items():
            record = records[evaluation_id]
            observed = (
                record.get("dataset_id"), record.get("partition_column"), record.get("fit_values"), record.get("coefficient_values"), record.get("scored_values"), record.get("reserved_values"), record.get("fold_indices"), record.get("winsorization"), record.get("winsor_quantile"), record.get("scored_prediction_aggregation"), record.get("expected_scored_rows"), record.get("coefficient_policy"), record.get("historical_neural_column"),
            )
            if observed != contract:
                errors.append(f"dataset-adapter contract mismatch: {evaluation_id}")
    parity = json.loads(ADAPTER_PARITY.read_text(encoding="utf-8"))
    if parity.get("status") != "PASS" or parity.get("gate") != "dataset_specific_adapter_a1" or parity.get("model_identity") != "aug2_head4":
        errors.append("dataset-adapter parity status is not PASS")
    if (parity.get("row_level_inputs_redistributed"), parity.get("embeddings_redistributed"), parity.get("predictions_redistributed")) != (False, False, False):
        errors.append("dataset-adapter parity manifest misstates redistribution")
    role_checks = parity.get("role_checks", [])
    if len(role_checks) != 26 or {record.get("evaluation_id") for record in role_checks} != set(expected):
        errors.append("dataset-adapter role parity coverage mismatch")
    if any(any(value != 0 for value in record.get("role_mismatches", [])) for record in role_checks):
        errors.append("dataset-adapter partition-role parity mismatch")
    preprocessing = parity.get("preprocessing_checks", [])
    aggregation = parity.get("aggregation_checks", [])
    if {record.get("evaluation_id") for record in preprocessing} != set(expected) or {record.get("evaluation_id") for record in aggregation} != set(expected):
        errors.append("dataset-adapter preprocessing or aggregation coverage mismatch")
    if parity.get("maximum_role_mismatches") != 0 or parity.get("maximum_preprocessing_absolute_difference") != 0.0:
        errors.append("dataset-adapter historical implementation parity mismatch")
    if parity.get("maximum_aggregation_absolute_difference", float("inf")) > 2e-14:
        errors.append("dataset-adapter scored-prediction aggregation parity mismatch")
    if parity.get("scored_metadata_accepts_labels") is not False or parity.get("all_primary_evaluations_covered") is not True:
        errors.append("dataset-adapter label firewall or evaluation coverage is misstated")
    smoke = json.loads(ADAPTER_CLEAN_SMOKE.read_text(encoding="utf-8"))
    if smoke.get("status") != "PASS" or smoke.get("tests_discovered") != 65 or smoke.get("adapter_tests") != 10:
        errors.append("dataset-adapter clean-environment smoke test failed")
    if smoke.get("failures") != 0 or smoke.get("errors") != 0 or smoke.get("skipped") != 0:
        errors.append("dataset-adapter clean-environment tests were not clean")
    if smoke.get("validator_counterexamples") != 3 or smoke.get("jchem_winsorization_drift_rejected") is not True or smoke.get("tdc_fit_role_drift_rejected") is not True or smoke.get("aggregation_parity_mismatch_rejected") is not True:
        errors.append("dataset-adapter validator counterexamples were not recorded")


def validate_evaluation_runner(errors: list[str]) -> None:
    acquisition = json.loads(DATA_ACQUISITION.read_text(encoding="utf-8"))
    if acquisition.get("schema_version") != 1 or acquisition.get("network_default") != "deny":
        errors.append("evaluation-runner acquisition registry is unsafe or unsupported")
        return
    expected_sources = {
        "aqsoldb": ("8e02b548fd9a78778ff89a5aa9a460d1a289cc3a", "upstream_data_marked_cc0_1_0", "manual_pinned_upstream_acquisition"),
        "llompart_aqsoldbc_dataset": ("2.0", "etalab_2_0", "user_supplied_local_package"),
        "complat": ("d4feda24b4bcb9efee9953605ba56af0e52dba4d", "readme_declares_mit_no_license_file_dataset_rights_not_established", "user_supplied_local_package"),
        "tdc_dataset": ("c310c35f27e3f506411018ac43d97b8ba23ca652", "software_mit_dataset_rights_separate", "user_supplied_local_package"),
        "jchem_dataset": ("3da09dc96acde6c5c51d784307559067864ca779", "repository_mit_dataset_included_no_separate_data_notice", "user_supplied_local_package"),
        "chemberta_zinc_base_v1": ("761d6a18cf99db371e0b43baf3e2d21b3e865a20", "no_explicit_licence_in_retrieved_metadata", "identifier_only_no_bundled_weights"),
    }
    sources = {record.get("source_id"): record for record in acquisition.get("resources", [])}
    if set(sources) != set(expected_sources):
        errors.append("evaluation-runner acquisition source set mismatch")
    else:
        for source_id, expected in expected_sources.items():
            record = sources[source_id]
            observed = (record.get("revision"), record.get("licence_status"), record.get("acquisition_mode"))
            if observed != expected or record.get("automatic_download") is not False:
                errors.append(f"evaluation-runner acquisition contract mismatch: {source_id}")
        chemberta = sources.get("chemberta_zinc_base_v1", {})
        if chemberta.get("revision_role") != "revision observed at the provenance audit; not an identifier recorded by the historical training run" or chemberta.get("revision_used_in_historical_run") is not None:
            errors.append("ChemBERTa audit-observed and historical revision semantics are conflated")
    expected_source_map = {"R01": "llompart_aqsoldbc_dataset", "R02": "complat", "R03": "tdc_dataset", "R04": "tdc_dataset", "R05": "jchem_dataset", "R09": "complat"}
    if acquisition.get("evaluation_source_map") != expected_source_map:
        errors.append("evaluation-runner acquisition source mapping mismatch")
    parity = json.loads(EVALUATION_RUNNER_PARITY.read_text(encoding="utf-8"))
    if parity.get("status") != "PASS" or parity.get("gate") != "acquisition_data_layout_evaluation_runner_e1" or parity.get("model_identity") != "aug2_head4":
        errors.append("evaluation-runner parity status is not PASS")
    if (parity.get("row_level_inputs_redistributed"), parity.get("predictions_redistributed"), parity.get("scored_labels_redistributed")) != (False, False, False):
        errors.append("evaluation-runner parity manifest misstates redistribution")
    records = {record.get("evaluation_id"): record for record in parity.get("evaluations", [])}
    expected_rows = {key: value[1] for key, value in EXPECTED_EVALUATIONS.items()}
    expected_coefficients = {
        "R01": {"0": 0.5, "1": 0.5, "2": 0.5, "3": 0.5, "4": 0.5},
        "R02": {"0": 0.35783701629836606, "1": 0.34719288633809187, "2": 0.3483525332542907, "3": 0.35889311589834877, "4": 0.34989160615768977},
        "R03": {"0": 0.5107147095759091, "1": 0.5309883523177864, "2": 0.513853188782415, "3": 0.4749956727808701, "4": 0.5096218180594146},
        "R04": {"0": 0.5079547955184467},
        "R05": {"1": 0.5042305832703751, "2": 0.4575169080107138, "3": 0.5181060446991516, "4": 0.44616494504197796, "5": 0.5754599425836471},
        "R09": {"0": 0.3524569312559862},
    }
    if set(records) != set(expected_rows) or sum(record.get("rows", 0) for record in records.values()) != 38228:
        errors.append("evaluation-runner parity coverage or row total mismatch")
    else:
        for evaluation_id, expected_rows_value in expected_rows.items():
            record = records[evaluation_id]
            if record.get("rows") != expected_rows_value or record.get("coefficient_values") != expected_coefficients[evaluation_id]:
                errors.append(f"evaluation-runner coefficient or row parity mismatch: {evaluation_id}")
            if record.get("package_assembly_scored_labels_content_accessed") is not False:
                errors.append(f"evaluation-runner assembly accessed scored labels: {evaluation_id}")
            if record.get("maximum_descriptor_row_difference", float("inf")) > 2e-14 or record.get("maximum_neural_row_difference", float("inf")) > 2e-14 or record.get("maximum_dlg_sol_row_difference", float("inf")) > 2e-7:
                errors.append(f"evaluation-runner row-level parity mismatch: {evaluation_id}")
            if record.get("maximum_metric_absolute_difference") != 0.0 or record.get("maximum_bootstrap_absolute_difference") != 0.0:
                errors.append(f"evaluation-runner score parity mismatch: {evaluation_id}")
    if parity.get("all_assembly_phases_avoided_scored_label_content") is not True:
        errors.append("evaluation-runner label-access boundary is misstated")
    if parity.get("maximum_descriptor_row_difference", float("inf")) > 2e-14 or parity.get("maximum_neural_row_difference", float("inf")) > 2e-14 or parity.get("maximum_dlg_sol_row_difference", float("inf")) > 2e-7:
        errors.append("evaluation-runner aggregate prediction parity mismatch")
    if parity.get("maximum_metric_absolute_difference") != 0.0 or parity.get("maximum_bootstrap_absolute_difference") != 0.0:
        errors.append("evaluation-runner aggregate score parity mismatch")
    smoke = json.loads(EVALUATION_RUNNER_CLEAN_SMOKE.read_text(encoding="utf-8"))
    if smoke.get("status") != "PASS" or smoke.get("tests_discovered") != 81 or smoke.get("evaluation_runner_tests") != 16:
        errors.append("evaluation-runner clean-environment smoke test failed")
    if smoke.get("failures") != 0 or smoke.get("errors") != 0 or smoke.get("skipped") != 0:
        errors.append("evaluation-runner clean-environment tests were not clean")
    counterexample_keys = (
        "automatic_download_drift_rejected",
        "coefficient_parity_drift_rejected",
        "prediction_parity_drift_rejected",
        "label_access_drift_rejected",
        "evaluation_set_drift_rejected",
    )
    if smoke.get("validator_counterexamples") != 5 or any(smoke.get(key) is not True for key in counterexample_keys):
        errors.append("evaluation-runner validator counterexamples were not recorded")


def validate_benchmark_input_contracts(errors: list[str]) -> None:
    payload = json.loads(BENCHMARK_INPUT_CONTRACTS.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("network_default") != "deny" or payload.get("redistribution_status") != "not_bundled_local_use_only":
        errors.append("benchmark-input registry is unsafe or unsupported")
        return
    expected_files = {
        "molecules": {"path": "molecules.csv", "columns": ["record_id", "smiles"]},
        "labels": {"path": "labels.csv", "columns": ["record_id", "logS"]},
        "partitions": {"path": "partitions.csv", "columns": ["record_id", "fold_index", "role"]},
        "derivation": {"path": "derivation.json"},
    }
    if payload.get("files") != expected_files:
        errors.append("benchmark-input file schema mismatch")
    expected = {
        "R01": ("aqsoldbc", "llompart_aqsoldbc_dataset", [0, 1, 2, 3, 4], ["fit", "reserved", "scored"], "one_held_out_prediction", 8047),
        "R02": ("complat", "complat", [0, 1, 2, 3, 4], ["fit", "scored"], "one_held_out_prediction", 17937),
        "R03": ("tdc", "tdc_dataset", [0, 1, 2, 3, 4], ["fit", "reserved", "scored"], "one_held_out_prediction", 7985),
        "R04": ("tdc", "tdc_dataset", [0, 1, 2, 3, 4], ["fit", "reserved", "scored"], "same_rows_in_every_fold", 1997),
        "R05": ("jchem", "jchem_dataset", [1, 2, 3, 4, 5], ["fit", "coefficient", "scored"], "same_rows_in_every_fold", 980),
        "R09": ("complat", "complat", [0], ["fit", "scored"], "single_full_refit", 1282),
    }
    records = {record.get("evaluation_id"): record for record in payload.get("evaluation_contracts", [])}
    if set(records) != set(expected):
        errors.append("benchmark-input evaluation set mismatch")
    else:
        for evaluation_id, contract in expected.items():
            record = records[evaluation_id]
            observed = (record.get("dataset_id"), record.get("upstream_source_id"), record.get("fold_indices"), record.get("required_roles"), record.get("scored_assignment"), record.get("expected_scored_rows"))
            if observed != contract:
                errors.append(f"benchmark-input evaluation contract mismatch: {evaluation_id}")
    evidence = payload.get("source_evidence", {})
    expected_evidence = {
        "llompart_aqsoldbc_dataset": ("pinned_dataverse_object_verified", [
            ("AqSolDBc.tab (file id 569036; original CSV download)", 2333803, "d41e512b43d66d838c8d3644df6ea99d37e0b60ee7705f0f1631da4d37011fc1"),
        ]),
        "tdc_dataset": ("pinned_historical_runtime_exports_verified", [
            ("TDC_AqSol_benchmark_train_val.csv", 334137, "6a93ef8265f03950a29e916a8e8a4878f9c3f6324bfe4e02cbf50493b0843d39"),
            ("TDC_AqSol_benchmark_test.csv", 98645, "f04dc12b97a5f3ee219e008f0eeadecf92beae88d429ff88e78faa3e47186b72"),
        ]),
        "complat": ("pinned_repository_objects_verified", [
            ("data/final_data/final_unique_train.csv", 2814578, "8f57b8e7ecf640f861f35bb17ff61b7b9c0592aa675ef8a186a429a4b6fab00f"),
            ("data/final_data/final_unique_test.csv", 138236, "5a34e2ab090fbabd48cea483f8d554e2cde33767b5a0acba9eb1c4f45395134a"),
        ]),
        "jchem_dataset": ("pinned_repository_objects_verified", [
            ("dataset.xlsx", 3132459, "416375f45fb58806a67b7124f5ab40b54faa50f329669338244c579fad6f4af1"),
        ]),
    }
    if set(evidence) != set(expected_evidence):
        errors.append("benchmark-input source evidence set mismatch")
    else:
        for source_id, (status, objects) in expected_evidence.items():
            record = evidence[source_id]
            observed_objects = [(item.get("path"), item.get("bytes"), item.get("sha256")) for item in record.get("objects", [])]
            if record.get("status") != status or observed_objects != objects:
                errors.append(f"benchmark-input source evidence mismatch: {source_id}")
    if sha256(AQSOLDBC_PARTITION_RECIPE) != "78e2dd21bcb96e633a1a4ce98606a2438a464649d0a8adb41b4592188a0690a9":
        errors.append("AqSolDBc partition recipe identity mismatch")
    reconstruction = json.loads(AQSOLDBC_RECONSTRUCTION.read_text(encoding="utf-8"))
    if reconstruction.get("status") != "PASS" or reconstruction.get("evaluation_id") != "R01":
        errors.append("AqSolDBc reconstruction audit failed")
    if reconstruction.get("recipe_sha256") != sha256(AQSOLDBC_PARTITION_RECIPE):
        errors.append("AqSolDBc reconstruction recipe binding mismatch")
    expected_outputs = {
        "molecules.csv": (8047, "4d03d61ccfacb1c596b189a3f55e3de77c04b6a8a6c5c0af72fac07bfcfbd12d"),
        "labels.csv": (8047, "4fdd3c2041e1283a6fa941e757eb1f856ca466e58e90e80c65c94171018ab7ea"),
        "partitions.csv": (40235, "9ab293b51dbcc6ffb1f5467d881f0dce7ed2744fc84f44e8b1556f1bae2393ac"),
    }
    generated = reconstruction.get("generated_files", {})
    for name, (rows, digest) in expected_outputs.items():
        record = generated.get(name, {})
        if record.get("rows") != rows or record.get("sha256") != digest:
            errors.append(f"AqSolDBc reconstructed output mismatch: {name}")
    identity = reconstruction.get("historical_identity_check", {})
    if identity.get("record_id_order_exact") is not True or identity.get("curated_smiles_exact") is not True:
        errors.append("AqSolDBc source-row identity mismatch")
    if identity.get("maximum_absolute_target_difference") != 0.0 or identity.get("outer_fold_mismatches") != 0:
        errors.append("AqSolDBc target or outer-fold identity mismatch")
    if identity.get("total_role_assignments_checked") != 40235 or set(identity.get("role_assignment_mismatches_by_fold", {}).values()) != {0}:
        errors.append("AqSolDBc fold-role reconstruction mismatch")

    recipe_hashes = {
        COMPLAT_PARTITION_RECIPE: "8178c562efaf70d3e47d344024da6b02f539763540bd528d92fa874ef84f4651",
        JCHEM_PARTITION_RECIPE: "66825dd97ac60439fe63c77ac58b8e254ed53537ba1ecd0bb522d2521856215b",
        TDC_PARTITION_RECIPE: "a48e800a2a27531763e54f47fe1dcb4f6af1dcbf0646471a586f72969366e184",
    }
    for path, expected_hash in recipe_hashes.items():
        if sha256(path) != expected_hash:
            errors.append(f"benchmark partition recipe identity mismatch: {path.name}")
    complat = json.loads(COMPLAT_RECONSTRUCTION.read_text(encoding="utf-8"))
    if complat.get("status") != "PASS" or complat.get("gate") != "complat_input_reconstruction_v1a3" or complat.get("row_level_inputs_redistributed") is not False:
        errors.append("ComPlat input reconstruction audit failed")
    if complat.get("recipe_sha256") != sha256(COMPLAT_PARTITION_RECIPE):
        errors.append("ComPlat reconstruction recipe binding mismatch")
    complat_expected = {
        "R02": (17937, 89685, {
            "molecules.csv": "24ce7d64d4a23f89ed6bc2d15efa29e9f732cc719930db53d60ba713b725647c",
            "labels.csv": "972357373e1f4aafa771425e6cc8a7b5f243825c3d771f4526e4295cf15a95d2",
            "partitions.csv": "917de99078e419268d2a6547c6bf9187b37f747e76ea6e177bc0896d743442c7",
        }),
        "R09": (19219, 19219, {
            "molecules.csv": "29bb40b676bdaa26aabbe98389244ea9d3fa273073936f1194903fd03364f1a3",
            "labels.csv": "ec385c069594c7fa5ba0aa0960e08db1141358062ae86e647bf9adbdaf67bb3c",
            "partitions.csv": "b086f406e015f4f2029caeb454ab83b523cabb1b77dcb6157340daa5f3ea7ceb",
        }),
    }
    for evaluation_id, (rows, partition_rows, digests) in complat_expected.items():
        record = complat.get("evaluations", {}).get(evaluation_id, {})
        if record.get("molecule_rows") != rows or record.get("partition_rows") != partition_rows or record.get("generated_sha256") != digests:
            errors.append(f"ComPlat reconstructed output mismatch: {evaluation_id}")
    r02_identity = complat.get("evaluations", {}).get("R02", {}).get("historical_identity", {})
    if r02_identity.get("record_id_order_exact") is not True or r02_identity.get("canonical_smiles_exact") is not True:
        errors.append("ComPlat R02 source-row identity mismatch")
    if r02_identity.get("maximum_absolute_target_difference", float("inf")) > 1e-12 or r02_identity.get("fold_assignment_mismatches") != 0 or r02_identity.get("role_assignments_checked") != 89685:
        errors.append("ComPlat R02 fold identity mismatch")
    r09_identity = complat.get("evaluations", {}).get("R09", {}).get("historical_identity", {})
    if r09_identity.get("test_record_id_order_exact") is not True or r09_identity.get("maximum_absolute_test_target_difference") != 0.0 or (r09_identity.get("fit_rows"), r09_identity.get("scored_rows")) != (17937, 1282):
        errors.append("ComPlat R09 test identity mismatch")
    jchem = json.loads(JCHEM_RECONSTRUCTION.read_text(encoding="utf-8"))
    if jchem.get("status") != "PASS" or jchem.get("gate") != "jchem_input_reconstruction_v1a3" or jchem.get("row_level_inputs_redistributed") is not False:
        errors.append("JCheM input reconstruction audit failed")
    if jchem.get("recipe_sha256") != sha256(JCHEM_PARTITION_RECIPE):
        errors.append("JCheM reconstruction recipe binding mismatch")
    jchem_expected = {
        "molecules.csv": (9798, "a516f75ec67b18a5fd0dcc165e512a4435bd5fd888ef72980731f94922942d06"),
        "labels.csv": (9798, "b409a952df03ca40eb52da60512a9bcb5d999a2466cc8476ff017cbabecde23c"),
        "partitions.csv": (48990, "fb24b2913aa8df83fb12fbe7e4775661e83e258392c995cc83487dcea5633d41"),
    }
    for name, (rows, digest) in jchem_expected.items():
        record = jchem.get("generated_files", {}).get(name, {})
        if record.get("rows") != rows or record.get("sha256") != digest:
            errors.append(f"JCheM reconstructed output mismatch: {name}")
    jchem_identity = jchem.get("historical_identity", {})
    jchem_identity_contract = (
        jchem_identity.get("valid_rows"),
        jchem_identity.get("invalid_source_rows"),
        jchem_identity.get("record_id_order_exact"),
        jchem_identity.get("canonical_smiles_exact"),
        jchem_identity.get("role_assignment_mismatches"),
        jchem_identity.get("role_assignments_checked"),
        jchem_identity.get("fixed_test_rows"),
        jchem_identity.get("fixed_test_membership_equal_across_five_splits"),
    )
    if jchem_identity_contract != (9798, [8340, 8483], True, True, 0, 48990, 980, True):
        errors.append("JCheM fold-role identity mismatch")
    if jchem_identity.get("maximum_test_label_rounding_delta", float("inf")) > 0.0050001:
        errors.append("JCheM fixed-test label rounding boundary mismatch")
    version_check = jchem.get("canonicalization_version_check", {})
    if version_check.get("pinned_reconstruction_version") != "RDKit 2026.03.1" or version_check.get("canonical_smiles_differences") != 3 or version_check.get("historical_output_reproduced_by_pinned_input_environment") is not True:
        errors.append("JCheM canonicalization-version boundary mismatch")
    tdc = json.loads(TDC_RECONSTRUCTION.read_text(encoding="utf-8"))
    if tdc.get("status") != "PASS" or tdc.get("gate") != "tdc_input_reconstruction_v1a4" or tdc.get("row_level_inputs_redistributed") is not False:
        errors.append("TDC input reconstruction audit failed")
    if tdc.get("recipe_sha256") != sha256(TDC_PARTITION_RECIPE):
        errors.append("TDC reconstruction recipe binding mismatch")
    tdc_expected = {
        "R03": {
            "molecules.csv": (9982, "b5b2dad6637d8e89193ce537f9d55fa2c3c84a3fd81e3a037d8bcbd0b6949eef"),
            "labels.csv": (9982, "0ec176e131b46b3fbfea9a0c29e868b13112325f55d3fefc8c7a15e2c4ceeaca"),
            "partitions.csv": (49910, "007788b7be1ea4dde5556ab62b893bd2926cbe0edcb094885062c54d369b3339"),
        },
        "R04": {
            "molecules.csv": (9982, "b5b2dad6637d8e89193ce537f9d55fa2c3c84a3fd81e3a037d8bcbd0b6949eef"),
            "labels.csv": (9982, "0ec176e131b46b3fbfea9a0c29e868b13112325f55d3fefc8c7a15e2c4ceeaca"),
            "partitions.csv": (49910, "b8bae932254bc50e03a2b991ac54f8f0ced5cbd00664befbe29e9aafc18c1cea"),
        },
    }
    for evaluation_id, files in tdc_expected.items():
        record = tdc.get("evaluations", {}).get(evaluation_id, {})
        if (record.get("molecule_rows"), record.get("partition_rows")) != (9982, 49910):
            errors.append(f"TDC reconstructed row-count mismatch: {evaluation_id}")
        for name, (rows, digest) in files.items():
            observed = record.get("generated_files", {}).get(name, {})
            if observed.get("rows") != rows or observed.get("sha256") != digest:
                errors.append(f"TDC reconstructed output mismatch: {evaluation_id} {name}")
        identity = record.get("historical_identity", {})
        if identity.get("train_canonical_smiles_exact") is not True or identity.get("test_canonical_smiles_exact") is not True:
            errors.append(f"TDC canonical structure identity mismatch: {evaluation_id}")
        if identity.get("native_role_assignment_mismatches") != 0 or identity.get("native_role_assignments_checked") != 49910:
            errors.append(f"TDC native fold-role identity mismatch: {evaluation_id}")
    r03_identity = tdc.get("evaluations", {}).get("R03", {}).get("historical_identity", {})
    if (r03_identity.get("locked_scored_rows"), r03_identity.get("locked_scored_record_id_exact"), r03_identity.get("locked_scored_fold_mismatches"), r03_identity.get("locked_scored_canonical_smiles_exact"), r03_identity.get("locked_scored_maximum_absolute_target_difference")) != (7985, True, 0, True, 0.0):
        errors.append("TDC R03 locked scored-row identity mismatch")
    r04_identity = tdc.get("evaluations", {}).get("R04", {}).get("historical_identity", {})
    if (r04_identity.get("locked_scored_rows"), r04_identity.get("locked_scored_record_id_order_exact"), r04_identity.get("locked_scored_maximum_absolute_target_difference"), r04_identity.get("scored_repetitions_per_test_row")) != (1997, True, 0.0, 5):
        errors.append("TDC R04 locked scored-row identity mismatch")
    source_identity = tdc.get("source", {}).get("historical_runtime_export_identity", {})
    if source_identity != {"train_raw_smiles_exact": True, "test_raw_smiles_exact": True, "maximum_absolute_target_difference": 0.0, "train_record_id_order_exact": True, "test_record_id_order_exact": True}:
        errors.append("TDC historical runtime-export identity mismatch")
    tdc_version = tdc.get("canonicalization_version_boundary", {})
    if tdc_version != {"pinned_reconstruction_version": "RDKit 2023.09.6", "later_comparison_version": "RDKit 2026.03.1", "later_version_train_canonical_differences": 80, "later_version_test_canonical_differences_including_parse_failures": 11, "later_version_test_parse_failures": 2, "historical_output_reproduced_by_pinned_tdc_environment": True}:
        errors.append("TDC canonicalization-version boundary mismatch")
    tdc_smoke = json.loads(TDC_INPUT_PREPARATION_CLEAN_SMOKE.read_text(encoding="utf-8"))
    tdc_environment = {"numpy": "1.26.4", "pandas": "2.2.3", "scikit_learn": "1.6.1", "rdkit": "2023.09.6"}
    if tdc_smoke.get("status") != "PASS" or tdc_smoke.get("gate") != "tdc_input_preparation_v1a4" or tdc_smoke.get("environment") != tdc_environment:
        errors.append("TDC input-preparation clean-environment smoke failed")
    if (tdc_smoke.get("tests_discovered"), tdc_smoke.get("failures"), tdc_smoke.get("errors"), tdc_smoke.get("skipped")) != (3, 0, 0, 0):
        errors.append("TDC input-preparation tests were not clean")
    if tdc_smoke.get("pinned_source_execution") != {"R03": "PASS", "R04": "PASS"} or tdc_smoke.get("generated_output_hashes_match_recipe") is not True or tdc_smoke.get("validated_packages") != {"R03": "PASS", "R04": "PASS"}:
        errors.append("TDC input-preparation pinned-source execution failed")
    smoke = json.loads(INPUT_PREPARATION_CLEAN_SMOKE.read_text(encoding="utf-8"))
    expected_environment = {"numpy": "1.26.4", "pandas": "2.2.3", "scikit_learn": "1.6.1", "openpyxl": "3.1.5", "rdkit": "2026.03.1"}
    if smoke.get("status") != "PASS" or smoke.get("gate") != "benchmark_input_preparation_v1a3" or smoke.get("environment") != expected_environment:
        errors.append("benchmark-input preparation clean-environment smoke failed")
    if (smoke.get("tests_discovered"), smoke.get("failures"), smoke.get("errors"), smoke.get("skipped")) != (6, 0, 0, 0):
        errors.append("benchmark-input preparation tests were not clean")
    if smoke.get("pinned_source_execution") != {"R02": "PASS", "R05": "PASS", "R09": "PASS"} or smoke.get("generated_output_hashes_match_recipes") is not True:
        errors.append("benchmark-input pinned-source execution failed")


def validate_training_role_contracts(errors: list[str]) -> None:
    contracts = json.loads(TRAINING_ROLE_CONTRACTS.read_text(encoding="utf-8"))
    parity = json.loads(TRAINING_ROLE_PARITY.read_text(encoding="utf-8"))
    expected_evaluations = set(EXPECTED_EVALUATIONS)
    expected_branches = {"descriptor", "language", "geometry", "fusion_head", "blend_coefficient"}
    if contracts.get("schema_version") != 1 or contracts.get("model_identity") != "aug2_head4":
        errors.append("training-role contract identity mismatch")
    if contracts.get("status") != "AUDITED_MATERIALIZER_AVAILABLE":
        errors.append("training-role contract overstates or understates its implementation status")
    evaluations = contracts.get("evaluations", {})
    if set(evaluations) != expected_evaluations:
        errors.append("training-role contract evaluation set mismatch")
        return
    unit_total = 0
    expected_unit_counts = {
        "R01": {branch: 5 for branch in expected_branches},
        "R02": {branch: 5 for branch in expected_branches},
        "R03": {branch: 5 for branch in expected_branches},
        "R04": {**{branch: 5 for branch in expected_branches - {"blend_coefficient"}}, "blend_coefficient": 1},
        "R05": {branch: 5 for branch in expected_branches},
        "R09": {branch: 1 for branch in expected_branches},
    }
    for evaluation_id, evaluation in evaluations.items():
        branches = evaluation.get("branches", {})
        if set(branches) != expected_branches:
            errors.append(f"training-role branch set mismatch: {evaluation_id}")
            continue
        for branch_id, branch in branches.items():
            if branch.get("scored_label_permission") != "scoring_only":
                errors.append(f"training-role scored-label permission mismatch: {evaluation_id}/{branch_id}")
            units = branch.get("expected_units", [])
            unit_total += len(units)
            if len(units) != expected_unit_counts[evaluation_id][branch_id]:
                errors.append(f"training-role unit-count mismatch: {evaluation_id}/{branch_id}")
            for unit in units:
                overlaps = unit.get("scored_label_overlap_rows", {})
                policy = unit.get("scored_overlap_policy")
                if policy == "zero_for_all_training_stages":
                    if any(overlaps.values()):
                        errors.append(f"unexpected scored-label overlap: {evaluation_id}/{branch_id}/{unit.get('fold_index')}")
                elif policy == "documented_non_nested_global_preselection":
                    if evaluation_id != "R02" or branch_id not in {"descriptor", "language", "geometry"}:
                        errors.append("training-role exception escaped its R02 scope")
                    if any(overlaps.get(stage) for stage in ("parameter_fit", "final_refit", "coefficient_fit")):
                        errors.append(f"R02 scored labels reached final fitting: {branch_id}/{unit.get('fold_index')}")
                else:
                    errors.append(f"unknown training-role overlap policy: {evaluation_id}/{branch_id}")
    if unit_total != 126:
        errors.append("training-role total unit count mismatch")
    if parity.get("schema_version") != 1 or parity.get("gate") != "training_role_contract_b0" or parity.get("status") != "PASS":
        errors.append("training-role parity status mismatch")
    if parity.get("contract_sha256") != sha256(TRAINING_ROLE_CONTRACTS):
        errors.append("training-role parity is not bound to the current contract")
    if parity.get("contract_units") != 126 or parity.get("unresolved_contracts") != 0:
        errors.append("training-role parity coverage mismatch")
    if parity.get("r02_exception_units") != 15:
        errors.append("training-role parity does not record every R02 selection exception")
    if parity.get("all_non_exception_scored_training_overlaps_zero") is not True:
        errors.append("training-role non-exception overlap audit failed")
    if parity.get("all_final_fit_scored_overlaps_zero") is not True:
        errors.append("training-role final-fit firewall audit failed")
    materialization = json.loads(TRAINING_ROLE_MATERIALIZATION_PARITY.read_text(encoding="utf-8"))
    if materialization.get("schema_version") != 1 or materialization.get("gate") != "training_role_materializer_b1" or materialization.get("status") != "PASS":
        errors.append("training-role materialization parity status mismatch")
    if materialization.get("model_identity") != "aug2_head4" or materialization.get("row_level_inputs_redistributed") is not False:
        errors.append("training-role materialization identity or redistribution status mismatch")
    if materialization.get("contract_sha256") != sha256(TRAINING_ROLE_CONTRACTS):
        errors.append("training-role materialization is not bound to the current contract")
    materializer_files = {
        "src/dlg_sol/workflow/materialization.py": ROOT / "src/dlg_sol/workflow/materialization.py",
        "scripts/materialize_training_unit.py": ROOT / "scripts/materialize_training_unit.py",
        "tests/test_training_role_materialization.py": ROOT / "tests/test_training_role_materialization.py",
    }
    implementation = materialization.get("implementation_sha256", {})
    if implementation != {name: sha256(path) for name, path in materializer_files.items()}:
        errors.append("training-role materialization implementation binding mismatch")
    if materialization.get("materialized_units") != 126 or materialization.get("nested_selection_units") != 20:
        errors.append("training-role materialization coverage mismatch")
    if materialization.get("r02_documented_exception_units") != 15:
        errors.append("training-role materialization does not retain every disclosed R02 exception")
    if materialization.get("all_stage_counts_and_record_id_hashes_match_contract") is not True:
        errors.append("training-role materialization row-identity parity failed")
    if materialization.get("all_scored_views_label_free") is not True:
        errors.append("training-role materialization scored-label firewall failed")
    dependencies = materialization.get("cross_evaluation_dependencies_verified", {})
    if dependencies != {"R04_coefficient_source": "complete_R03_OOF", "R09_coefficient_source": "complete_R02_OOF"}:
        errors.append("training-role materialization dependency contract mismatch")


def validate_training_execution_recipes(errors: list[str]) -> None:
    recipes = json.loads(TRAINING_EXECUTION_RECIPES.read_text(encoding="utf-8"))
    roles = json.loads(TRAINING_ROLE_CONTRACTS.read_text(encoding="utf-8"))
    if recipes.get("schema_version") != 1 or recipes.get("model_identity") != "aug2_head4":
        errors.append("training-execution recipe identity mismatch")
        return
    if recipes.get("status") != "AUDITED_EXECUTION_RECIPES":
        errors.append("training-execution recipe status mismatch")
    if recipes.get("training_role_contract") != {
        "path": "configs/training_role_contracts.json",
        "reference_method": "evaluation_id_branch_id_and_all_declared_folds",
        "duplicates_row_counts_or_hashes": False,
    }:
        errors.append("training-execution recipes duplicate or misreference row contracts")
    output = recipes.get("output_contract", {})
    if output != {
        "prediction_columns": ["record_id", "prediction"],
        "scored_labels_permitted": False,
        "run_manifest_required": True,
        "private_absolute_paths_permitted": False,
        "historical_weight_identity_claimed": False,
        "historical_metric_identity_claimed": False,
    }:
        errors.append("training-execution output contract mismatch")
    scope = recipes.get("scope", {})
    if scope != {
        "unit_level_branch_execution": True,
        "single_command_six_evaluation_orchestration": False,
        "automatic_data_acquisition": False,
        "gpu_cluster_scheduling": False,
        "fresh_training_is_protocol_reproduction": True,
    }:
        errors.append("training-execution scope mismatch")
    evaluations = recipes.get("evaluations", {})
    role_evaluations = roles.get("evaluations", {})
    expected_branches = {"descriptor", "language", "geometry", "fusion_head", "blend_coefficient"}
    if set(evaluations) != set(role_evaluations):
        errors.append("training-execution evaluation coverage mismatch")
        return
    recipe_count = 0
    referenced_units = 0
    forbidden_keys = {"rows", "row_count", "record_id_sha256", "stage_assignment_sha256"}
    configuration_paths = set()
    for evaluation_id, records in evaluations.items():
        if set(records) != expected_branches:
            errors.append(f"training-execution branch coverage mismatch: {evaluation_id}")
            continue
        for branch_id, record in records.items():
            recipe_count += 1
            reference = record.get("unit_reference")
            if reference != {"evaluation_id": evaluation_id, "branch_id": branch_id, "folds": "all_declared"}:
                errors.append(f"training-execution unit reference mismatch: {evaluation_id}/{branch_id}")
            if forbidden_keys & set(record):
                errors.append(f"training-execution row contract duplication: {evaluation_id}/{branch_id}")
            units = role_evaluations[evaluation_id]["branches"][branch_id]["expected_units"]
            referenced_units += len(units)
            expected_stages = [set(unit.get("stages", {})) for unit in units]
            if any(set(record.get("input_stages", [])) != stages for stages in expected_stages):
                errors.append(f"training-execution stage mismatch: {evaluation_id}/{branch_id}")
            for reference_path in record.get("configuration_refs", []):
                relative = str(reference_path).partition("#")[0]
                configuration_paths.add(relative)
                if not (ROOT / relative).is_file():
                    errors.append(f"training-execution configuration reference missing: {relative}")
    if recipe_count != 30 or referenced_units != 126:
        errors.append("training-execution recipe or unit coverage mismatch")
    for branch_id in ("descriptor", "language", "geometry"):
        r02 = evaluations["R02"][branch_id]
        if r02.get("selection_mode") != "frozen_global_preselection":
            errors.append(f"R02 primary recipe permits fold-local reselection: {branch_id}")
        r09 = evaluations["R09"][branch_id]
        if r09.get("selection_mode") != "reuse_r02_frozen_global_preselection":
            errors.append(f"R09 representation selection provenance mismatch: {branch_id}")
    for branch_id in ("descriptor", "language", "geometry", "fusion_head"):
        r04 = evaluations["R04"][branch_id]
        dependency = {"evaluation_id": "R03", "branch_id": branch_id, "fold_mapping": "same_fold"}
        if r04.get("execution_type") != "reuse_companion_model" or r04.get("dependency") != dependency:
            errors.append(f"R04 companion-model reuse mismatch: {branch_id}")
    parity = json.loads(TRAINING_EXECUTION_PARITY.read_text(encoding="utf-8"))
    if parity.get("schema_version") != 1 or parity.get("gate") != "training_execution_recipe_b2" or parity.get("status") != "PASS":
        errors.append("training-execution parity status mismatch")
        return
    if parity.get("model_identity") != "aug2_head4" or parity.get("row_level_inputs_redistributed") is not False:
        errors.append("training-execution parity identity or redistribution mismatch")
    if parity.get("recipe_sha256") != sha256(TRAINING_EXECUTION_RECIPES) or parity.get("training_role_contract_sha256") != sha256(TRAINING_ROLE_CONTRACTS):
        errors.append("training-execution parity binding mismatch")
    implementation_paths = {
        "src/dlg_sol/workflow/execution.py": ROOT / "src/dlg_sol/workflow/execution.py",
        "tests/test_training_execution_recipes.py": ROOT / "tests/test_training_execution_recipes.py",
    }
    if parity.get("implementation_sha256") != {name: sha256(path) for name, path in implementation_paths.items()}:
        errors.append("training-execution implementation binding mismatch")
    expected_configuration_hashes = {name: sha256(ROOT / name) for name in sorted(configuration_paths)}
    if parity.get("configuration_reference_sha256") != expected_configuration_hashes:
        errors.append("training-execution configuration-reference binding mismatch")
    if parity.get("recipe_count") != 30 or parity.get("referenced_training_units") != 126:
        errors.append("training-execution parity coverage mismatch")
    if parity.get("row_contract_counts_or_hashes_duplicated") is not False:
        errors.append("training-execution parity permits duplicate row contracts")
    complat = parity.get("complat_global_preselection_reconstruction", {})
    if complat.get("source_sha256") != "8f57b8e7ecf640f861f35bb17ff61b7b9c0592aa675ef8a186a429a4b6fab00f":
        errors.append("ComPlat global-preselection source identity mismatch")
    expected_complat = {
        "selection_fit": (14351, "3bac17bf2fbe92b263fe9ddad8d8cbd865ec8077a3ae3d8a79575b68217a37e3"),
        "selection_score": (3586, "6f63362f7cd7433dbd1cbb79993a4ec6ab8b06c581493515ffa645eaf5c89f3d"),
    }
    for stage, (rows, digest) in expected_complat.items():
        observed = complat.get(stage, {})
        if (observed.get("rows"), observed.get("record_id_sha256"), observed.get("historical_membership_mismatches")) != (rows, digest, 0):
            errors.append(f"ComPlat global-preselection reconstruction mismatch: {stage}")
    if complat.get("rdkit") != "2023.09.6" or complat.get("contracts_checked") != 18 or complat.get("all_r02_r09_representation_contracts_match") is not True:
        errors.append("ComPlat global-preselection contract coverage mismatch")
    if parity.get("r02_primary_selection_mode") != "frozen_global_preselection" or parity.get("r02_fold_local_reselection_is_primary") is not False:
        errors.append("R02 primary selection mode is misstated")


def validate_branch_execution(errors: list[str]) -> None:
    scripts = {
        "scripts/train_descriptor_unit.py",
        "scripts/train_language_unit.py",
        "scripts/train_geometry_unit.py",
        "scripts/train_fusion_unit.py",
    }
    for relative in scripts:
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"missing branch execution entry point: {relative}")
            continue
        source = path.read_text(encoding="utf-8")
        if '"--materialized"' not in source or '"--output"' not in source:
            errors.append(f"branch entry point omits the materialized input contract: {relative}")
        if '"--labels"' in source or '"--benchmark"' in source:
            errors.append(f"branch entry point accepts a prohibited direct label or benchmark input: {relative}")
    language = json.loads(LANGUAGE_VARIANTS.read_text(encoding="utf-8")).get("execution_profiles", {})
    geometry = json.loads(GEOMETRY_CONFIG.read_text(encoding="utf-8")).get("execution_profiles", {})
    expected_ids = set(EXPECTED_EVALUATIONS)
    if set(language) != expected_ids or set(geometry) != expected_ids:
        errors.append("branch execution profiles do not cover all primary evaluations")
    else:
        if language["R02"].get("selected_epoch") != 14 or language["R09"].get("selected_epoch") != 14:
            errors.append("ComPlat ChemBERTa frozen epoch contract mismatch")
        if language["R03"].get("seed_base") != 260762 or language["R03"].get("fold_seed_multiplier") != 100:
            errors.append("TDC ChemBERTa seed schedule mismatch")
        if geometry["R01"].get("selected_trial") != 2:
            errors.append("AqSolDBc geometry transferred-trial contract mismatch")
        if (geometry["R02"].get("selected_trial"), geometry["R02"].get("best_epoch")) != (0, 19):
            errors.append("ComPlat geometry frozen-setting contract mismatch")
        if (geometry["R03"].get("candidate_count"), geometry["R03"].get("maximum_epochs"), geometry["R03"].get("patience")) != (4, 140, 20):
            errors.append("TDC geometry search contract mismatch")
        if geometry["R05"].get("parameters") != geometry["R01"].get("parameters"):
            errors.append("JCheM geometry transferred-parameter contract mismatch")
    record = json.loads(BRANCH_EXECUTION_INTEGRATION.read_text(encoding="utf-8"))
    if record.get("schema_version") != 1 or record.get("gate") != "training_branch_execution_b3" or record.get("status") != "PASS":
        errors.append("branch execution integration status mismatch")
        return
    if set(record.get("entry_points", [])) != scripts:
        errors.append("branch execution integration entry-point coverage mismatch")
    boundary = record.get("input_boundary", {})
    if boundary.get("required_primary_input") != "materialized training-unit directory" or boundary.get("original_benchmark_table_accepted_by_branch_cli") is not False or boundary.get("scored_labels_permitted") is not False:
        errors.append("branch execution input boundary is unsafe")
    run = record.get("user_local_descriptor_integration", {})
    expected = {
        "evaluation_id": "R05",
        "fold_index": 1,
        "benchmark_rows": 9798,
        "parameter_fit_rows": 6860,
        "selection_score_rows": 1958,
        "scored_prediction_rows": 980,
        "retained_descriptor_columns": 571,
        "selected_trial": 3,
        "recorded_selected_trial": 3,
        "prediction_rows": 980,
        "all_predictions_finite": True,
        "scored_labels_accessed": False,
        "descriptor_recomputation_exercised": False,
    }
    if any(run.get(key) != value for key, value in expected.items()):
        errors.append("branch execution descriptor integration mismatch")
    if run.get("prediction_columns") != ["record_id", "prediction"]:
        errors.append("branch execution prediction schema mismatch")
    if record.get("redistributed_row_level_inputs") is not False or record.get("redistributed_predictions") is not False or record.get("complete_six_evaluation_fresh_training_claimed") is not False:
        errors.append("branch execution integration overstates the release scope")


def validate_release_provenance(errors: list[str]) -> None:
    external_adapter = json.loads(EXTERNAL_ADAPTER_PROVENANCE.read_text(encoding="utf-8"))
    redistribution = json.loads(REDISTRIBUTION_DECISIONS.read_text(encoding="utf-8"))
    with TABLE_4.open(encoding="utf-8", newline="") as handle:
        table_records = list(csv.DictReader(handle))
    errors.extend(validate_provenance_contract(external_adapter, redistribution, table_records))

    sources = {
        record.get("id"): record
        for record in json.loads(EXTERNAL_SOURCES.read_text(encoding="utf-8")).get("sources", [])
    }
    family_source_ids = {family.get("source_id") for family in external_adapter.get("families", [])}
    if not family_source_ids.issubset(sources):
        errors.append("external-adapter source is absent from the external-source registry")
    chemberta = sources.get("chemberta_zinc_base_v1", {})
    if chemberta.get("revision_observed_at_audit") != "761d6a18cf99db371e0b43baf3e2d21b3e865a20":
        errors.append("ChemBERTa audit-observed revision mismatch")
    if chemberta.get("revision_used_in_historical_run") is not None or chemberta.get("revision_status") != "immutable revision was not recorded in the historical run":
        errors.append("ChemBERTa historical revision status mismatch")

    external_audit = json.loads(EXTERNAL_ADAPTER_PROVENANCE_AUDIT.read_text(encoding="utf-8"))
    expected_external = {
        "status": "PASS",
        "families": 4,
        "model_records": 5,
        "historical_files_checked": 18,
        "historical_files_found": 18,
        "hash_mismatches": 0,
        "repository_revisions_checked": 4,
        "repository_revision_mismatches": 0,
        "execution_environments_recorded": 4,
        "evidence_class_mismatches": 0,
        "third_party_source_bundled": False,
        "study_adapter_wrapper_bundled": False,
        "row_level_prediction_bundled": False,
        "private_paths_recorded": False,
    }
    if any(external_audit.get(key) != value for key, value in expected_external.items()):
        errors.append("external-adapter provenance audit mismatch")
    expected_main_text_external = {
        "pnnl_gnn:none",
        "ali_xgb125:none",
        "bhattacharya_roy:no_interaction",
        "ulrich_consensus_adaptation:retrained",
        "ulrich_consensus:released",
    }
    if set(external_audit.get("main_text_external_models_covered", [])) != expected_main_text_external:
        errors.append("external-adapter main-text model coverage mismatch")

    redistribution_audit = json.loads(REDISTRIBUTION_DECISION_AUDIT.read_text(encoding="utf-8"))
    expected_redistribution = {
        "status": "PASS",
        "source_decisions": 8,
        "sources_without_decision": 0,
        "row_level_dataset_sources_bundled": 0,
        "excluded_sensitive_derived_asset_classes": 5,
        "included_public_asset_classes": 2,
        "licence_finding_and_bundling_decision_separated": True,
        "machine_readable_supplementary_results_retained": True,
        "independent_row_level_metric_recalculation_without_upstream_inputs_possible": False,
        "private_paths_recorded": False,
    }
    if any(redistribution_audit.get(key) != value for key, value in expected_redistribution.items()):
        errors.append("redistribution-decision audit mismatch")

    data_files = {path.relative_to(ROOT).as_posix() for path in (ROOT / "data").rglob("*") if path.is_file()}
    if data_files != {"data/README.md"}:
        errors.append("row-level or undeclared data file is present in the public data directory")
    prohibited_roots = {"checkpoints", "weights", "third_party", "raw_data"}
    if any((ROOT / name).exists() for name in prohibited_roots):
        errors.append("excluded third-party or model-asset directory is present")

    v1c = json.loads(V1C_RESULT_REGENERATION_AUDIT.read_text(encoding="utf-8"))
    if (
        v1c.get("status") != "PASS"
        or v1c.get("public_result_regeneration_scope_complete") is not True
        or v1c.get("fresh_six_evaluation_retraining_claimed") is not False
        or v1c.get("table_3_source_context_status") != "PASS"
        or v1c.get("table_4_regeneration_status") != "PASS"
        or v1c.get("article_table_3_contract_status") != "PASS"
        or v1c.get("article_table_4_contract_status") != "PASS"
    ):
        errors.append("V1-C result-regeneration status mismatch")
    if v1c.get("machine_readable_supplementary_tables") != 14 or v1c.get("metric_parity_maximum_absolute_difference") != 0.0 or v1c.get("bootstrap_parity_maximum_absolute_difference") != 0.0:
        errors.append("V1-C result-regeneration evidence mismatch")
    required_artefacts = v1c.get("required_artefacts", [])
    if len(required_artefacts) != 8:
        errors.append("V1-C required-artefact coverage mismatch")
    for artefact in required_artefacts:
        path = ROOT / artefact.get("path", "")
        if not path.is_file() or sha256(path) != artefact.get("sha256"):
            errors.append(f"V1-C artefact identity mismatch: {artefact.get('path')}")


def scan_public_text(errors: list[str]) -> None:
    for path in ROOT.rglob("*"):
        if (
            not path.is_file()
            or ".git" in path.parts
            or (
                path.name not in TEXT_DOTFILES
                and path.suffix.lower() not in TEXT_SUFFIXES
            )
        ):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        relative = path.relative_to(ROOT).as_posix()
        for label, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(relative):
                errors.append(f"{label} in path: {relative}")
            if pattern.search(text):
                errors.append(f"{label}: {relative}")


def main() -> int:
    errors: list[str] = []
    validate_manifest(errors)
    validate_release_policy(errors)
    validate_tables(errors)
    validate_weighting_summary(errors)
    validate_article_table_3(errors)
    validate_article_table_4(errors)
    validate_environment_spec(errors)
    validate_evaluation_registry(errors)
    validate_core_parity(errors)
    validate_descriptor_branch(errors)
    validate_training_orchestration(errors)
    validate_language_branch(errors)
    validate_geometry_branch(errors)
    validate_fusion_branch(errors)
    validate_dataset_adapters(errors)
    validate_evaluation_runner(errors)
    validate_benchmark_input_contracts(errors)
    validate_training_role_contracts(errors)
    validate_training_execution_recipes(errors)
    validate_branch_execution(errors)
    validate_release_provenance(errors)
    scan_public_text(errors)
    if errors:
        print("STAGING_VALIDATION=FAIL")
        for error in sorted(set(errors)):
            print(f"- {error}")
        return 1
    print("STAGING_VALIDATION=PASS")
    print("GITHUB_RELEASE_PACKAGE_READY=YES")
    print("PUBLIC_RELEASE_READY=YES")
    return 0


if __name__ == "__main__":
    sys.exit(main())
