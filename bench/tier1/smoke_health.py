#!/usr/bin/env python3
"""Tier-1 smoke: verify vLLM API is reachable."""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    for path in ("/health", "/v1/models"):
        url = f"{base}{path}"
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                body = resp.read(4096).decode("utf-8", errors="replace")
            print(f"OK {url} ({resp.status}) {body[:120]}...")
        except urllib.error.HTTPError as exc:
            print(f"HTTP {exc.code} {url}")
            if exc.code >= 500:
                return 1
        except Exception as exc:
            print(f"FAIL {url}: {exc}")
            return 1

    print("tier1 smoke_health: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
