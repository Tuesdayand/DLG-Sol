#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "release_manifest.json"
EXCLUDED_PARTS = {".git", "__pycache__"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def included_files() -> list[Path]:
    paths = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path == OUTPUT:
            continue
        if any(part in EXCLUDED_PARTS for part in path.relative_to(ROOT).parts):
            continue
        paths.append(path)
    return sorted(paths, key=lambda item: item.relative_to(ROOT).as_posix())


def main() -> None:
    files = {}
    for path in included_files():
        relative = path.relative_to(ROOT).as_posix()
        files[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    payload = {
        "schema_version": 1,
        "release_stage": "V1_REPRODUCIBILITY_PACKAGE_RELEASE",
        "files": files,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT} with {len(files)} files")


if __name__ == "__main__":
    main()
