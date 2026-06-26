#!/usr/bin/env bash
# Recreate pri-inference from ghcr.io/punkrecords/inference:dev with host pri/ overlay.
set -euo pipefail

PRI="${PRI:-/home/wasnaga/punk-records-inference}"
MODEL_MOUNT="${MODEL_MOUNT:-/home/wasnaga/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B-FP8/snapshots/0b2752837483aa34b3db6e83e151b150c0e00e49}"
IMAGE="${PRI_IMAGE:-ghcr.io/punkrecords/inference:dev}"

sed -i 's/\r$//' "${PRI}/docker/start.sh" 2>/dev/null || true
chmod +x "${PRI}/docker/start.sh"

docker rm -f pri-inference 2>/dev/null || true

docker create --name pri-inference \
  --gpus all \
  -p 8000:8000 \
  -v "${MODEL_MOUNT}:/model:ro" \
  -v pri-data:/data/pri \
  -e GPU_MEMORY_UTILIZATION=0.60 \
  -e MODEL_PATH=/model \
  -e NLS_MODEL_PATH=/model \
  -e NLS_AGENT_SHIM=1 \
  -e NLS_CHAIN_CAPTURE_MODE=turn \
  --entrypoint /opt/pri-repo/docker/start.sh \
  "${IMAGE}"

echo "[deploy] overlay pri/ + start.sh from ${PRI}"
docker cp "${PRI}/pri/." pri-inference:/opt/pri-repo/pri/
docker cp "${PRI}/docker/start.sh" pri-inference:/opt/pri-repo/docker/start.sh
docker start pri-inference

echo "[deploy] waiting for /health (up to 8 min)..."
for i in $(seq 1 96); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "HEALTH_OK"
    docker logs pri-inference 2>&1 | grep -E 'PRIMicroscope|microscope patch|Uvicorn running' | tail -5 || true
    exit 0
  fi
  sleep 5
done

echo "HEALTH_TIMEOUT — last logs:"
docker logs pri-inference 2>&1 | tail -25
exit 1
