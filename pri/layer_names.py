"""Map vLLM module paths to transformer layer indices (plug-and-play).

vLLM keys in ``forward_context.no_compile_layers`` and ``kv_cache_groups``
vary by ``--model-impl`` and architecture family:

| Backend / family        | Example layer key              |
|-------------------------|--------------------------------|
| Classic vLLM            | ``model.layers.12.self_attn``  |
| Transformers (Llama 3)  | ``12.attn``                    |
| Decoder wrappers        | ``decoder.layers.3.self_attn`` |

PRI capture readback and K/V inject index tensors by integer layer id.
A single regex for ``layers.N.`` silently skips whole models (e.g. Llama 70B
on ``--model-impl transformers``) → zero captures and dead RESUME inject.
"""

from __future__ import annotations

import re

# Most specific patterns first; first match wins.
_LAYER_INDEX_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)"),
    re.compile(r"^(\d+)\.attn(?:\.|$)"),
    re.compile(r"^decoder\.layers\.(\d+)\."),
    re.compile(r"^model\.layers\.(\d+)\."),
)


def extract_layer_index(layer_name: str) -> int | None:
    """Return zero-based layer index from a vLLM layer / KV-cache module name."""
    if not layer_name:
        return None
    for pattern in _LAYER_INDEX_PATTERNS:
        match = pattern.search(layer_name)
        if match:
            return int(match.group(1))
    return None


def classify_layer_names(layer_names: list[str]) -> dict[str, list[str]]:
    """Partition names into mapped vs unmapped (for boot diagnostics)."""
    mapped: list[str] = []
    unmapped: list[str] = []
    for name in layer_names:
        if extract_layer_index(name) is None:
            unmapped.append(name)
        else:
            mapped.append(name)
    return {"mapped": mapped, "unmapped": unmapped}
