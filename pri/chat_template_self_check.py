#!/usr/bin/env python3
"""Probe capture_start for the live vLLM model (plug-and-play template check).

Usage:
    python3 -m pri.chat_template_self_check --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import sys

import requests

from pri.chat_template import run_capture_start_self_check


def _resolve_model(base_url: str) -> str:
    response = requests.get(f"{base_url.rstrip('/')}/v1/models", timeout=15)
    response.raise_for_status()
    models = response.json().get("data") or []
    if not models:
        raise RuntimeError("no models from /v1/models")
    return str(models[0]["id"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Chat-template capture_start self-check")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="vLLM base URL",
    )
    parser.add_argument(
        "--system-prompt",
        default=(
            "You are a personal assistant with persistent memory. "
            "Answer from prior conversation context when available."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON report")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    model = _resolve_model(base)
    report = run_capture_start_self_check(
        api_root=base,
        model=model,
        system_prompt=args.system_prompt,
    )
    report["model"] = model
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(
            f"model={model} capture_start={report['capture_start']} "
            f"ok={report['ok']}",
        )
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
