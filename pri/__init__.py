"""Punk Records Inference — vLLM KV capture/resume plugin.

This package implements the PRI plugin loaded by vLLM via ``--kv-transfer-config``:

- **Capture** — serialize turn KV + hybrid state to ``.nls`` manifests on disk
- **Resume** — inject prior chain blocks so the model skips re-prefill
- **Retrieve** — optional Swiss semantic retrieval for overflow profiles

Entry modules: ``connector`` (vLLM hook), ``middleware.agent_shim`` (agent clients),
``admin`` (debug API). See ``docs/ARCHITECTURE.md`` for the full layout.
"""

__version__ = "0.0.1-dev"