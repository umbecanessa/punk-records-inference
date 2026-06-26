#!/usr/bin/env bash
# Source bench/.env (gitignored) for OPENROUTER_API_KEY etc.
# Safe to source from other bench scripts.

_BENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_ENV_FILE="${_BENCH_ROOT}/bench/.env"

if [[ -f "$_ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$_ENV_FILE"
  set +a
fi

unset _BENCH_ROOT _ENV_FILE
