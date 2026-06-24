#!/usr/bin/env python3
"""Collect Phase C storage/profile artifacts into bench/results run folder."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import requests

_ROOT = Path(__file__).resolve().parent.parent


def _du_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    if path.is_file():
        return path.stat().st_size
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def _capture_rows(captures_dir: Path) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    if not captures_dir.is_dir():
        return rows
    for nls in sorted(captures_dir.glob("*.nls")):
        rows.append(
            {
                "file": nls.name,
                "bytes": nls.stat().st_size,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect store stats for bench run")
    parser.add_argument("--base-url", default=os.environ.get("PRI_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--user-id", default="", help="Optional user_id for admin stats filter")
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("NLS_MEMORY_DIR", "/data/pri"),
        help="PRI data root (profile + captures)",
    )
    parser.add_argument("--out", required=True, help="Output store_stats.json path")
    parser.add_argument(
        "--capture-sizes-csv",
        default="",
        help="Optional capture_sizes.csv path (defaults next to --out)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    csv_path = Path(args.capture_sizes_csv) if args.capture_sizes_csv else out_path.with_name("capture_sizes.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    admin_stats: dict | None = None
    try:
        params = {"user_id": args.user_id} if args.user_id else {}
        r = requests.get(
            f"{args.base_url.rstrip('/')}/admin/memory/stats",
            params=params,
            timeout=30,
        )
        r.raise_for_status()
        admin_stats = r.json()
    except Exception as exc:
        admin_stats = {"error": str(exc)}

    profile_json: dict | str | None = None
    profile_path = data_dir / "model_profile.json"
    if profile_path.is_file():
        try:
            profile_json = json.loads(profile_path.read_text(encoding="utf-8"))
        except Exception as exc:
            profile_json = {"error": str(exc)}

    profile_env = ""
    profile_env_path = data_dir / "profile.env"
    if profile_env_path.is_file():
        profile_env = profile_env_path.read_text(encoding="utf-8", errors="replace")

    captures_dir = data_dir / "snapshot" / "captures"
    capture_rows = _capture_rows(captures_dir)

    index_rows = 0
    index_path = data_dir / "index.jsonl"
    if index_path.is_file():
        index_rows = sum(1 for _ in index_path.open(encoding="utf-8", errors="replace"))

    git_sha: str | None = None
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True
        ).strip()
    except Exception:
        pass

    payload = {
        "base_url": args.base_url,
        "data_dir": str(data_dir),
        "git_sha": git_sha,
        "admin_stats": admin_stats,
        "profile_json": profile_json,
        "profile_env": profile_env,
        "index_row_count": index_rows,
        "capture_count": len(capture_rows),
        "capture_bytes_total": sum(int(r["bytes"]) for r in capture_rows),
        "data_dir_bytes": _du_bytes(data_dir),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "bytes"])
        writer.writeheader()
        writer.writerows(capture_rows)

    print(f"store_stats -> {out_path}")
    print(f"capture_sizes -> {csv_path} ({len(capture_rows)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
