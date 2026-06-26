#!/usr/bin/env python3
"""Quick classic resume vs force-inject comparison."""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

_TIER1 = Path(__file__).resolve().parent
_REPO = _TIER1.parents[1]
for path in (_REPO, _REPO / "bench" / "opencode", _TIER1):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import requests

from chain_helpers import fetch_user_memories, select_chain_latest, TURN_ROLES
from nls_kvp_helpers import api_root_from_chat_url, enrich_kv_params
from sweep_lib import RECALL, SYSTEM_PROMPT, resolve_model, score_recall_clean, strip_think

BASE_URL = "http://127.0.0.1:8000"
UID = "hr_sweep_12c6137747"
CHAIN = "chain_thread_eceebeab8bfc"


def run(label: str, kv: dict) -> None:
    api = f"{BASE_URL.rstrip('/')}/v1/chat/completions"
    model = resolve_model(BASE_URL)
    q, exp = RECALL[3]
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ],
        "max_tokens": 160,
        "temperature": 0.0,
        "kv_transfer_params": kv,
        "chat_template_kwargs": {"enable_thinking": False},
        "cache_salt": f"c_{label}_{uuid.uuid4().hex[:6]}",
    }
    r = requests.post(api, json=body, timeout=300)
    if r.status_code >= 400:
        print(label, "HTTP", r.status_code, r.text[:120])
        return
    text = strip_think(r.json()["choices"][0]["message"]["content"] or "")
    scored = score_recall_clean(text, exp)
    status = "PASS" if scored["pass"] else "FAIL"
    print(label, status, repr(text[:110]))


def main() -> int:
    api_root = api_root_from_chat_url(BASE_URL)
    model = resolve_model(BASE_URL)
    mems = fetch_user_memories(api_root, UID, include_kv=True, limit=500)
    blocks = select_chain_latest(mems, CHAIN, k=0, max_tokens=0, roles=TURN_ROLES)
    snaps = [
        {"path": b["kvPath"], "num_tokens": int(b["numTokens"] or 0)}
        for b in blocks
        if b.get("kvPath")
    ]
    print(f"classic blocks={len(snaps)} tok={sum(s['num_tokens'] for s in snaps)}")

    run(
        "force_full",
        enrich_kv_params(
            {
                "memory_user": "force",
                "memory_ring": "general",
                "memory_no_capture": "1",
                "memory_force_inject": json.dumps(snaps),
                "memory_inject_layout": "resume",
            },
            SYSTEM_PROMPT,
            api_root=api_root,
            model=model,
        ),
    )
    run(
        "resume_no_enrich",
        {
            "memory_user": UID,
            "memory_ring": "general",
            "memory_no_capture": "1",
            "memory_inject_mode": "resume",
            "memory_base_session": CHAIN,
            "memory_resume_max_tokens": "28000",
        },
    )
    kv = enrich_kv_params(
        {
            "memory_user": UID,
            "memory_ring": "general",
            "memory_no_capture": "1",
            "memory_inject_mode": "resume",
            "memory_base_session": CHAIN,
            "memory_resume_max_tokens": "28000",
        },
        SYSTEM_PROMPT,
        api_root=api_root,
        model=model,
    )
    print("sys_hash", kv.get("memory_sys_prompt_hash", "")[:20])
    run("resume_enrich", kv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
