#!/usr/bin/env bash
# Wait for Llama 3.3 70B FP8 download, deploy, wipe, run cp20 turn_sweep.
set -uo pipefail

PRI="${PRI:-/home/wasnaga/punk-records-inference}"
OUT="${PRI}/bench/results/model_matrix_llama70b/pass1"
LOG="${OUT}/llama_cp20_pipeline.log"
SWEEP="${OUT}/turn_sweep_cp20.json"
DOWNLOAD_LOG="${HOME}/llama70b_download.log"
HF_REPO="${HF_REPO:-RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic}"

mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1

echo "[llama_cp20] $(date -Iseconds) pipeline start"

# Wait for download (up to 6h)
for i in $(seq 1 2160); do
  if grep -q "^DONE" "$DOWNLOAD_LOG" 2>/dev/null; then
    echo "[llama_cp20] download complete"
    break
  fi
  if ! pgrep -f "snapshot_download" >/dev/null 2>&1; then
    if python3 -c "from huggingface_hub import snapshot_download; snapshot_download('${HF_REPO}', local_files_only=True)" 2>/dev/null; then
      echo "[llama_cp20] model cached (no active download)"
      break
    fi
  fi
  if (( i % 60 == 0 )); then
    echo "[llama_cp20] waiting for download... ($(tail -1 "$DOWNLOAD_LOG" 2>/dev/null | head -c 120))"
  fi
  sleep 10
done

python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('${HF_REPO}'))" || {
  echo "[llama_cp20] FATAL: model not cached"
  exit 1
}

chmod +x "${PRI}/bench/deploy_swap_gx10.sh" "${PRI}/bench/deploy_model_gx10.sh" "${PRI}/docker/start.sh" 2>/dev/null || true
sed -i 's/\r$//' "${PRI}/bench/deploy_swap_gx10.sh" "${PRI}/bench/deploy_model_gx10.sh" "${PRI}/docker/start.sh" 2>/dev/null || true

echo "[llama_cp20] deploy swap (stops pri-gemma + all pri-* before 70B boot)"
export HF_REPO
"${PRI}/bench/deploy_swap_gx10.sh" llama || { echo "FATAL deploy"; exit 1; }

echo "[llama_cp20] overlay pri + harness"
docker cp "${PRI}/pri/." pri-llama:/opt/pri-repo/pri/
docker cp "${PRI}/bench/tier1/." pri-llama:/opt/pri-repo/bench/tier1/ 2>/dev/null || true
docker cp "${PRI}/bench/opencode/nls_kvp_helpers.py" pri-llama:/opt/pri-repo/bench/opencode/nls_kvp_helpers.py 2>/dev/null || true

PYTHONPATH="${PRI}" python3 -m pri.chat_template_self_check --base-url http://127.0.0.1:8000 || true

echo "[llama_cp20] wipe memory"
env PRI_CONTAINER=pri-llama PRI_DATA_VOLUME=pri-data-llama70b \
  "${PRI}/bench/wipe_memory.sh" --restart
for i in $(seq 1 144); do
  curl -sf http://127.0.0.1:8000/v1/models >/dev/null && break
  sleep 5
done
curl -sf http://127.0.0.1:8000/v1/models >/dev/null || { echo "FATAL vllm"; exit 1; }

# Re-overlay after restart
docker cp "${PRI}/pri/." pri-llama:/opt/pri-repo/pri/

echo "[llama_cp20] turn_sweep cp20"
PYTHONPATH="${PRI}" python3 -u "${PRI}/bench/tier1/turn_sweep.py" \
  --base-url http://127.0.0.1:8000 \
  --checkpoints 20 \
  --garbled-retries 2 \
  --no-stop-on-noise-garble \
  --out "$SWEEP"

echo "[llama_cp20] $(date -Iseconds) done"
python3 -c "
import json
d=json.load(open('${SWEEP}'))
r=d['results'][0]
print('TEXT', r['text_pass_clean'], '/5 RESUME', r['resume_pass_clean'], '/5 inject', r.get('turn_tokens'))
"
