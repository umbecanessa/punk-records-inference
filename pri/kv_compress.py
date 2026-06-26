"""Int8 KV tensor compression and zstd packaging for ``.nls`` payloads.

Symmetric per-tensor int8 quantization (max-abs / 127) plus zstd level 1.
Typical ratio ~3× vs bf16 on hybrid checkpoints with fast decompress on inject.

Used by ``pri.format`` when writing capture blobs. Legacy ``.kvz`` standalone
format uses the same ``torch.save({tensors, scales, meta})`` envelope.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Union

import torch

logger = logging.getLogger("nls_kv_compress")

try:
    import zstandard as zstd
    _HAS_ZSTD = True
except ImportError:
    _HAS_ZSTD = False
    logger.warning("zstandard not installed — .kvz files will use zlib fallback")

if not _HAS_ZSTD:
    import zlib

ZSTD_LEVEL = 1


def save_compressed(data: dict, path: Union[str, Path]) -> int:
    """Save a KV snapshot dict as int8+zstd compressed .kvz file.

    Returns the compressed file size in bytes.
    """
    path = Path(path)
    tensors_int8 = {}
    scales = {}
    meta = {}

    for k, v in data.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            vf = v.float()
            scale = vf.abs().max().item() / 127.0
            if scale < 1e-10:
                scale = 1.0
            q = (vf / scale).round().clamp(-127, 127).to(torch.int8)
            tensors_int8[k] = q
            scales[k] = scale
        elif isinstance(v, torch.Tensor):
            meta[k] = v
        else:
            meta[k] = v

    buf = io.BytesIO()
    torch.save({"tensors": tensors_int8, "scales": scales, "meta": meta}, buf)
    raw = buf.getvalue()

    if _HAS_ZSTD:
        cctx = zstd.ZstdCompressor(level=ZSTD_LEVEL)
        compressed = cctx.compress(raw)
    else:
        compressed = zlib.compress(raw, 1)

    path.write_bytes(compressed)
    return len(compressed)


def load_compressed(path: Union[str, Path]) -> dict:
    """Load a .kvz file back to the original dict with bf16/fp16 tensors.

    Dequantizes int8 → float32, then casts to the original precision
    (bfloat16, matching the model's KV dtype).
    """
    path = Path(path)
    compressed = path.read_bytes()

    if _HAS_ZSTD:
        dctx = zstd.ZstdDecompressor()
        raw = dctx.decompress(compressed)
    else:
        raw = zlib.decompress(compressed)

    buf = io.BytesIO(raw)
    saved = torch.load(buf, map_location="cpu", weights_only=False)

    tensors_int8 = saved["tensors"]
    scales = saved["scales"]
    meta = saved.get("meta", {})

    result = {}
    for k, q in tensors_int8.items():
        scale = scales[k]
        result[k] = (q.float() * scale).to(torch.bfloat16)

    for k, v in meta.items():
        result[k] = v

    return result
