#!/usr/bin/env python3
"""Write a turn-2+ .nls manifest sidecar with rope_start > 0 (KL #648 proof).

Uses nls_kvp_helpers (same contract as hosted openai.service.ts) to send
memory_capture_start on the wire. Run against a live vLLM instance:

  python bench/opencode/manifest_proof.py --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bench.opencode.nls_kvp_helpers import api_root_from_chat_url, enrich_kv_params
from pri.format import read_manifest

OUT_PATH = _ROOT / "bench" / "results" / "manifest_opencode_t2.json"
SYSTEM = (
    "You are OpenCode, an autonomous coding agent operating inside the user's "
    "workspace. Be concise."
)


def _model_id(base_url: str) -> str:
    r = requests.get(f"{base_url.rstrip('/')}/v1/models", timeout=30)
    r.raise_for_status()
    return r.json()["data"][0]["id"]


def _post(api: str, body: dict) -> None:
    r = requests.post(api, json=body, timeout=300)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
    data = r.json()
    if not data or not data.get("choices"):
        raise RuntimeError(f"empty completion: {r.text[:200]!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Manifest proof (rope_start > 0)")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="vLLM base URL (no /v1 suffix)",
    )
    parser.add_argument(
        "--captures-dir",
        default="",
        help="Host path to snapshot/captures (optional; skip file lookup if empty)",
    )
    parser.add_argument(
        "--out",
        default=str(OUT_PATH),
        help="JSON sidecar output path",
    )
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    api = f"{base}/v1/chat/completions"
    api_root = api_root_from_chat_url(api)
    model = _model_id(base)
    uid = "manifest_proof"
    chain = "chain_manifest_kl648"

    kvp1 = enrich_kv_params(
        {
            "memory_user": uid,
            "memory_ring": "general",
            "memory_base_session": chain,
            "memory_session": f"{chain}_t1_user",
            "memory_turn_index": "1",
            "memory_block_role": "user",
            "memory_text": "Plant fact: backend port is 4242.",
            "memory_silo": "1",
        },
        SYSTEM,
        api_root=api_root,
        model=model,
    )
    _post(
        api,
        {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Plant fact: backend port is 4242."},
            ],
            "max_tokens": 40,
            "temperature": 0,
            "user": uid,
            "kv_transfer_params": kvp1,
            "stream": False,
        },
    )
    time.sleep(2)

    kvp2 = enrich_kv_params(
        {
            "memory_user": uid,
            "memory_base_session": chain,
            "memory_session": f"{chain}_t2_user",
            "memory_turn_index": "2",
            "memory_block_role": "user",
            "memory_text": "What backend port did I specify?",
            "memory_inject_mode": "resume",
            "memory_deltanet_init_session": f"{chain}_t1_user",
        },
        SYSTEM,
        api_root=api_root,
        model=model,
    )
    _post(
        api,
        {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "What backend port did I specify?"},
            ],
            "max_tokens": 40,
            "temperature": 0,
            "user": uid,
            "kv_transfer_params": kvp2,
            "stream": False,
        },
    )
    time.sleep(2)

    manifest: dict | None = None
    nls_path: Path | None = None
    session_t2 = f"{chain}_t2_user"

    if args.captures_dir:
        cap_dir = Path(args.captures_dir)
        for p in sorted(cap_dir.glob("*.nls"), key=lambda x: x.stat().st_mtime, reverse=True):
            m = read_manifest(str(p))
            if m.get("session_id") == session_t2 and int(m.get("turn_index", 0)) >= 2:
                manifest = m
                nls_path = p
                break

    sidecar = {
        "chain_base": chain,
        "session_id_t2": session_t2,
        "memory_capture_start_t2": kvp2.get("memory_capture_start"),
        "memory_sys_prompt_hash": kvp2.get("memory_sys_prompt_hash"),
        "manifest": manifest,
        "nls_path": str(nls_path) if nls_path else None,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(json.dumps(sidecar, indent=2))

    if manifest is None:
        print("WARN: no .nls on disk — sidecar has kvp fields only", file=sys.stderr)
        return 0 if kvp2.get("memory_capture_start") else 1

    rope_start = int(manifest.get("rope_start", 0))
    turn_index = int(manifest.get("turn_index", 0))
    if rope_start <= 0 or turn_index < 2:
        print(
            f"FAIL: rope_start={rope_start} turn_index={turn_index}",
            file=sys.stderr,
        )
        return 1
    print(f"PASS: rope_start={rope_start} turn_index={turn_index}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
