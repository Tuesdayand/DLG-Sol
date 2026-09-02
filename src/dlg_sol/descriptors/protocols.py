from __future__ import annotations

import json
from pathlib import Path

from .filtering import DescriptorFilterConfig


DEFAULT_DESCRIPTOR_PROTOCOLS = Path(__file__).resolve().parents[3] / "configs" / "descriptor_protocols.json"


def load_descriptor_protocols(path: str | Path = DEFAULT_DESCRIPTOR_PROTOCOLS) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported descriptor-protocol schema")
    protocols = payload.get("protocols", {})
    if set(protocols) != {"aqsoldbc", "tdc", "complat", "jchem"}:
        raise ValueError("descriptor protocol set is incomplete")
    return payload


def filter_config_for(protocol: str, path: str | Path = DEFAULT_DESCRIPTOR_PROTOCOLS) -> DescriptorFilterConfig:
    payload = load_descriptor_protocols(path)
    if protocol not in payload["protocols"]:
        raise ValueError("unsupported descriptor protocol")
    values = payload["protocols"][protocol]["filtering"]
    return DescriptorFilterConfig(
        variance_threshold=float(values["variance_threshold"]),
        variance_ddof=int(values["variance_ddof"]),
        correlation_threshold=None if values["correlation_threshold"] is None else float(values["correlation_threshold"]),
        correlation_comparison=values["correlation_comparison"],
    )
