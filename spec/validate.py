#!/usr/bin/env python3
"""Validate .nls file headers against spec/manifest.schema.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pri.format import MAGIC, read_manifest

_SCHEMA_PATH = Path(__file__).resolve().parent / "manifest.schema.json"


def _load_schema() -> dict:
    with open(_SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)


def validate_manifest(manifest: dict, *, schema: dict | None = None) -> list[str]:
    """Return a list of validation errors (empty if valid)."""
    schema = schema or _load_schema()
    errors: list[str] = []

    if not isinstance(manifest, dict):
        return ["manifest is not an object"]

    required = schema.get("required", [])
    for key in required:
        if key not in manifest:
            errors.append(f"missing required field: {key}")

    props = schema.get("properties", {})
    for key, spec in props.items():
        if key not in manifest:
            continue
        val = manifest[key]
        expected = spec.get("type")
        if expected == "integer" and not isinstance(val, int):
            errors.append(f"{key}: expected integer, got {type(val).__name__}")
        elif expected == "number" and not isinstance(val, (int, float)):
            errors.append(f"{key}: expected number, got {type(val).__name__}")
        elif expected == "string" and not isinstance(val, str):
            errors.append(f"{key}: expected string, got {type(val).__name__}")
        elif expected == "boolean" and not isinstance(val, bool):
            errors.append(f"{key}: expected boolean, got {type(val).__name__}")
        elif expected == "array" and not isinstance(val, list):
            errors.append(f"{key}: expected array, got {type(val).__name__}")

        if key == "rope_end" and "rope_start" in manifest:
            if isinstance(val, int) and manifest["rope_start"] > val:
                errors.append("rope_start must be <= rope_end")

    return errors


def validate_nls_file(path: Path) -> list[str]:
    path = Path(path)
    if not path.is_file():
        return [f"file not found: {path}"]

    try:
        with open(path, "rb") as f:
            magic = f.read(4)
    except OSError as exc:
        return [str(exc)]

    if magic != MAGIC:
        return [f"invalid magic: {magic!r} (expected NLS\\x01)"]

    manifest = read_manifest(path)
    if manifest is None:
        return ["could not parse manifest JSON"]

    return validate_manifest(manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate .nls manifest headers")
    parser.add_argument("paths", nargs="+", help=".nls files or directories")
    args = parser.parse_args(argv)

    files: list[Path] = []
    for raw in args.paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(p.rglob("*.nls")))
        else:
            files.append(p)

    if not files:
        print("No .nls files found", file=sys.stderr)
        return 1

    failed = 0
    for fp in files:
        errs = validate_nls_file(fp)
        if errs:
            failed += 1
            print(f"FAIL {fp}")
            for e in errs:
                print(f"  - {e}")
        else:
            print(f"OK   {fp}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
