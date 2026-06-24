#!/usr/bin/env bash
# Wipe .nls captures and memory index on a running or stopped pri-inference box.
# Keeps model profile artifacts (/data/pri/profile.env, model_profile.json).
#
# Usage (on GX10 host):
#   ./bench/wipe_memory.sh
#   ./bench/wipe_memory.sh --restart   # wipe then docker restart pri-inference

set -euo pipefail

RESTART=0
CONTAINER="${PRI_CONTAINER:-pri-inference}"
VOLUME="${PRI_DATA_VOLUME:-pri-data}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --restart) RESTART=1; shift ;;
    -h|--help)
      echo "Usage: $0 [--restart]"
      exit 0
      ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "[wipe_memory] container=$CONTAINER volume=$VOLUME"

if docker inspect "$CONTAINER" >/dev/null 2>&1; then
  docker stop "$CONTAINER" >/dev/null || true
fi

docker run --rm -v "${VOLUME}:/data/pri" alpine:3.20 sh -c '
  set -e
  rm -rf /data/pri/snapshot/captures/* 2>/dev/null || true
  rm -f /data/pri/index.json /data/pri/index.jsonl 2>/dev/null || true
  rm -f /data/pri/snapshot/index.json /data/pri/snapshot/index.jsonl 2>/dev/null || true
  rm -f /data/pri/bm25_data.jsonl /data/pri/retrieval_log.jsonl 2>/dev/null || true
  rm -rf /data/pri/sc_embeddings/* 2>/dev/null || true
  rm -f /data/pri/fingerprints.npy /data/pri/semantic_embeddings_t1.npy 2>/dev/null || true
  rm -f /data/pri/delta_signal.npy /data/pri/delta_energy.npy /data/pri/delta_fingerprints_meta.json 2>/dev/null || true
  find /data/pri -name "*.nls" -delete 2>/dev/null || true
  echo "remaining under /data/pri:"
  find /data/pri -maxdepth 2 -type f 2>/dev/null | head -20 || true
'

if [[ "$RESTART" -eq 1 ]]; then
  docker start "$CONTAINER" >/dev/null
  echo "[wipe_memory] restarted $CONTAINER — wait for /health before benching"
else
  docker start "$CONTAINER" >/dev/null 2>&1 || true
fi

echo "[wipe_memory] done"
