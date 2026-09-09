from __future__ import annotations

from typing import Any, Iterable, Mapping


EXPECTED_FAMILIES = {"pnnl_gnn", "ali_xgb125", "bhattacharya_roy", "consensus_gnn"}
EXPECTED_COMPARATOR_RECORDS = {
    ("pnnl_gnn", "none"): "independently retrained adaptation",
    ("ali_xgb125", "none"): "split-adapted schema/settings; ComPlat same-split reproduction",
    ("bhattacharya_roy", "no_interaction"): "independently retrained adaptation",
    ("ulrich_consensus_adaptation", "retrained"): "independently retrained official-code adaptation",
    ("ulrich_consensus", "released"): "authors' released predictions",
}
EXPECTED_TABLE_4_RECORDS = {
    key: value
    for key, value in EXPECTED_COMPARATOR_RECORDS.items()
    if key != ("ulrich_consensus", "released")
}
EXPECTED_ADDITIONAL_RECORDS = {
    ("bhattacharya_roy", "interaction"): "prespecified architecture sensitivity",
}
EXPECTED_REDISTRIBUTION_SOURCES = {
    "llompart_aqsoldbc_dataset",
    "complat",
    "tdc_dataset",
    "jchem_dataset",
    "aqsoldb",
    "biogen_clnd",
    "ghanavati_aqueous_solubility_prediction",
    "chemberta_zinc_base_v1",
}
EXPECTED_DERIVED_DECISIONS = {
    "molecular_rows_and_experimental_labels": False,
    "source_linked_component_and_final_predictions": False,
    "language_and_geometry_embeddings": False,
    "conformer_coordinates_and_graph_caches": False,
    "historical_checkpoints": False,
    "aggregate_and_configuration_level_results": True,
    "original_dlg_sol_code_and_documentation": True,
}


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


def validate_provenance_contract(
    external_adapter: Mapping[str, Any],
    redistribution: Mapping[str, Any],
    table_records: Iterable[Mapping[str, Any]],
) -> list[str]:
    errors: list[str] = []
    families = external_adapter.get("families", [])
    by_family = {item.get("family_id"): item for item in families}
    if len(by_family) != len(families) or set(by_family) != EXPECTED_FAMILIES:
        errors.append("external comparator family coverage mismatch")
    observed_records: dict[tuple[str, str], str] = {}
    evidence_file_ids: list[str] = []
    for family in families:
        if family.get("bundled") is not False:
            errors.append("third-party comparator family is marked as bundled")
        if not family.get("execution_environment"):
            errors.append("external comparator execution environment is missing")
        for file_record in family.get("file_evidence", []):
            evidence_file_ids.append(str(file_record.get("file_id", "")))
            if file_record.get("bundled") is not False:
                errors.append("external comparator evidence file is marked as bundled")
            digest = file_record.get("sha256")
            byte_count = file_record.get("bytes")
            if (
                not file_record.get("file_id")
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or not isinstance(byte_count, int)
                or byte_count <= 0
            ):
                errors.append("external comparator evidence identity is incomplete")
        for record in family.get("model_records", []):
            key = (record.get("model_id"), record.get("variant"))
            observed_records[key] = record.get("evidence_class")
    if len(evidence_file_ids) != 18 or len(set(evidence_file_ids)) != 18:
        errors.append("external comparator evidence-file coverage mismatch")
    for key, expected in EXPECTED_COMPARATOR_RECORDS.items():
        if observed_records.get(key) != expected:
            errors.append(f"external comparator evidence class mismatch: {key[0]}:{key[1]}")
    for key, expected in EXPECTED_ADDITIONAL_RECORDS.items():
        if observed_records.get(key) != expected:
            errors.append(f"external comparator sensitivity class mismatch: {key[0]}:{key[1]}")
    observed_table = {
        (str(row["model_id"]), str(row["variant"])): str(row["evidence_class"])
        for row in table_records
        if str(row.get("model_id")) != "dlg_sol"
    }
    if observed_table != EXPECTED_TABLE_4_RECORDS:
        errors.append("Table 4 external comparator provenance mismatch")

    sources = redistribution.get("sources", [])
    by_source = {item.get("source_id"): item for item in sources}
    if len(by_source) != len(sources) or set(by_source) != EXPECTED_REDISTRIBUTION_SOURCES:
        errors.append("dataset redistribution decision coverage mismatch")
    for source in sources:
        if source.get("row_level_bundled") is not False:
            errors.append("row-level third-party source is marked as bundled")
        if not source.get("licence_finding") or not source.get("decision_reason"):
            errors.append("dataset redistribution rationale is incomplete")
    derived = {
        item.get("asset_class"): item.get("bundled")
        for item in redistribution.get("derived_asset_classes", [])
    }
    if derived != EXPECTED_DERIVED_DECISIONS:
        errors.append("derived-asset redistribution decision mismatch")

    forbidden = (
        "/data/" + "koo/",
        "/home/" + "koo/",
        "agent" + "_work",
        "code" + "x",
        "clau" + "de",
    )
    lowered = [value.lower() for value in _walk_strings([external_adapter, redistribution])]
    if any(marker in value for marker in forbidden for value in lowered):
        errors.append("private path or internal development name in provenance registry")
    return errors
