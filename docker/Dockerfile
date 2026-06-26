# Punk Records Inference — pinned vLLM + KV plugin
#
# Build:
#   docker build -f docker/Dockerfile -t ghcr.io/punkrecords/inference:dev .
#
# Run (BYOC — mount your checkpoint at /model):
#   docker compose -f docker/compose.yaml up

ARG VLLM_BASE=vllm/vllm-openai:cu130-nightly@sha256:a20a9fc7e06830cfe2050e8d18d76d0f7abc4755d44408cf9cc60749f4cf7c09
FROM ${VLLM_BASE}

LABEL org.opencontainers.image.source=https://github.com/umbecanessa/punk-records-inference
LABEL org.opencontainers.image.description="Punk Records Inference — KV capture/resume for vLLM"

ENV PYTHONPATH=/opt/pri-repo
ENV NLS_MEMORY_DIR=/data/pri
ENV NLS_SNAPSHOT_DIR=/data/pri/snapshot
ENV NLS_AGENT_SHIM=1
ENV NLS_CHAIN_CAPTURE_MODE=turn

WORKDIR /opt/pri-repo

COPY patches/ /opt/pri-repo/patches/
COPY pri/ /opt/pri-repo/pri/
COPY spec/ /opt/pri-repo/spec/
COPY docker/start.sh /opt/pri-repo/docker/start.sh
COPY bench/run_suite.sh /opt/pri-repo/bench/run_suite.sh

RUN chmod +x /opt/pri-repo/docker/start.sh /opt/pri-repo/bench/run_suite.sh && \
    sed -i 's/\r$//' /opt/pri-repo/docker/start.sh /opt/pri-repo/bench/run_suite.sh && \
    (python3 /opt/pri-repo/patches/apply_patches.py || \
     echo "[Dockerfile] WARN: apply_patches.py failed — image may need manual vLLM patch")

RUN pip install --no-cache-dir zstandard sentence-transformers

EXPOSE 8000

ENTRYPOINT ["/opt/pri-repo/docker/start.sh"]
