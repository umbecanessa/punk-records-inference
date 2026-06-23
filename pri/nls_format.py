"""NLS Snapshot Format — Binary format for hybrid Mamba+Attention state.

File layout (.nls):
  [4 bytes]  Magic: b"NLS\\x01"
  [2 bytes]  Format version (uint16 LE) — currently 1
  [4 bytes]  Manifest length in bytes (uint32 LE)
  [N bytes]  JSON manifest (UTF-8)
  [rest]     zstd-compressed tensor payload (int8 quantized + scales)

The manifest is readable without decompressing tensors — useful for
indexing, browsing, and debugging snapshots without loading GPU data.

Tensor payload is torch.save({
    "tensors": {key: int8_tensor},
    "scales":  {key: float32_scale},
    "meta":    {key: non-tensor value (e.g. _meta_seq_len)},
}) compressed with zstd level 1.

Quantization: per-tensor symmetric int8 (max-abs / 127).
Proven in KL #458: 42 MB bf16 -> 13.2 MB (3.2x), 22 ms decompress.
"""

from __future__ import annotations

import io
import json
import logging
import struct
import time
from pathlib import Path
from typing import Union

import torch

logger = logging.getLogger("nls_format")

MAGIC = b"NLS\x01"
FORMAT_VERSION = 1

try:
    import zstandard as zstd
    _HAS_ZSTD = True
except ImportError:
    _HAS_ZSTD = False
    logger.warning("zstandard not installed — .nls files will use zlib fallback")

if not _HAS_ZSTD:
    import zlib

ZSTD_LEVEL = 1


def _build_manifest(data: dict) -> dict:
    """Extract a JSON-serializable manifest from the tensor dict."""
    seq_len = 0
    has_mamba = 0
    attn_layers = []
    mamba_layers = []

    for k, v in data.items():
        if k == "_meta_seq_len" and isinstance(v, torch.Tensor):
            seq_len = v.item()
        elif k == "_meta_has_mamba" and isinstance(v, torch.Tensor):
            has_mamba = v.item()
        elif k.endswith("_k"):
            layer_idx = int(k.split("_")[1])
            if layer_idx not in attn_layers:
                attn_layers.append(layer_idx)
        elif "_mamba_conv" in k:
            layer_idx = int(k.split("_")[1])
            if layer_idx not in mamba_layers:
                mamba_layers.append(layer_idx)

    return {
        "version": FORMAT_VERSION,
        "seq_len": seq_len,
        "has_mamba": has_mamba,
        "attn_layers": sorted(attn_layers),
        "mamba_layers": sorted(mamba_layers),
        "num_keys": len(data),
        "created_at": time.time(),
    }


def save_nls(data: dict, path: Union[str, Path], extra_manifest: dict | None = None) -> int:
    """Save a KV snapshot dict as .nls file. Returns compressed file size."""
    path = Path(path)

    manifest = _build_manifest(data)
    if extra_manifest:
        manifest.update(extra_manifest)
    manifest_bytes = json.dumps(manifest, separators=(",", ":")).encode("utf-8")

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

    # NLS v2: compute block_hash from compressed payload (content address)
    import hashlib
    block_hash = hashlib.sha256(compressed).hexdigest()
    manifest["block_hash"] = block_hash
    # Expose block_hash to caller via the dict they passed in. This is what
    # `snapshot_connector._readback_and_save` reads at registration time so
    # the Memory entry's content-address matches the on-disk manifest.
    if extra_manifest is not None:
        extra_manifest["block_hash"] = block_hash

    # Re-encode manifest with block_hash included
    manifest_bytes = json.dumps(manifest, separators=(",", ":")).encode("utf-8")

    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<H", FORMAT_VERSION))
        f.write(struct.pack("<I", len(manifest_bytes)))
        f.write(manifest_bytes)
        f.write(compressed)

    return path.stat().st_size


def read_manifest(path: Union[str, Path]) -> dict | None:
    """Read only the JSON manifest from a .nls file (no tensor decompression)."""
    path = Path(path)
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != MAGIC:
                return None
            _version = struct.unpack("<H", f.read(2))[0]
            manifest_len = struct.unpack("<I", f.read(4))[0]
            manifest_bytes = f.read(manifest_len)
        return json.loads(manifest_bytes.decode("utf-8"))
    except Exception:
        return None


def load_nls(path: Union[str, Path]) -> dict:
    """Load a .nls file back to the original dict with bf16 tensors."""
    path = Path(path)
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"Not an NLS file: {path} (magic={magic!r})")
        _version = struct.unpack("<H", f.read(2))[0]
        manifest_len = struct.unpack("<I", f.read(4))[0]
        f.seek(manifest_len, 1)  # skip manifest
        compressed = f.read()

    import zlib as _zlib
    raw = None
    _zstd_mod = None
    if _HAS_ZSTD:
        _zstd_mod = zstd
    else:
        try:
            import zstandard as _zstd_mod
        except ImportError:
            pass
    if _zstd_mod is not None:
        try:
            dctx = _zstd_mod.ZstdDecompressor()
            raw = dctx.decompress(compressed)
        except Exception:
            pass
    if raw is None:
        raw = _zlib.decompress(compressed)

    buf = io.BytesIO(raw)
    saved = torch.load(buf, map_location="cpu", weights_only=False)

    tensors_int8 = saved["tensors"]
    scales_dict = saved["scales"]
    meta = saved.get("meta", {})

    result = {}
    for k, q in tensors_int8.items():
        result[k] = (q.float() * scales_dict[k]).to(torch.bfloat16)

    for k, v in meta.items():
        result[k] = v

    return result
