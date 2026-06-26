"""Microscope capture for PRI — layer hidden states + DeltaNet SSM during prefill.

Enable via ``kv_transfer_params`` on chat completions::

    {"microscope": "/tmp/nls_microscope", "microscope_tag": "inline_full"}

Loads with::

    --logits-processors pri.microscope_processor:PRIMicroscopeProcessor
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import torch
from vllm.config import VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor

logger = logging.getLogger("pri.microscope")

FULL_ATTN_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]

_microscope_enabled: bool = False
_microscope_capture: dict[str, torch.Tensor] = {}
_microscope_save_path: str = "/tmp/nls_microscope"
_microscope_tag: str = "capture"
_microscope_attn_idx: int = 0
_attn_patched: bool = False
_deltanet_patched: bool = False


def enable_microscope(save_path: str = "/tmp/nls_microscope", tag: str = "capture") -> None:
    global _microscope_enabled, _microscope_capture, _microscope_save_path, _microscope_tag
    _microscope_enabled = True
    _microscope_capture = {}
    _microscope_save_path = save_path
    _microscope_tag = tag
    os.makedirs(save_path, exist_ok=True)
    logger.info("MICROSCOPE ENABLED: tag=%s path=%s", tag, save_path)


def disable_microscope() -> None:
    global _microscope_enabled
    _microscope_enabled = False


def flush_microscope(tag: str = "capture") -> str | None:
    global _microscope_capture
    if not _microscope_capture:
        return None
    ts = int(time.time() * 1000)
    filepath = os.path.join(_microscope_save_path, f"microscope_{tag}_{ts}.pt")
    torch.save(_microscope_capture, filepath)
    n_keys = len(_microscope_capture)
    _microscope_capture = {}
    logger.info("MICROSCOPE flushed: %d tensors -> %s", n_keys, filepath)
    return filepath


def microscope_capture_hidden(
    layer_idx: int,
    hidden_states: torch.Tensor,
    *,
    stage: str = "attn_input",
) -> None:
    if not _microscope_enabled:
        return
    key = f"L{layer_idx}_{stage}_hs"
    _microscope_capture[key] = hidden_states.detach().cpu().clone()


def microscope_capture_deltanet(layer_idx: int, ssm_state: torch.Tensor) -> None:
    if not _microscope_enabled:
        return
    key = f"L{layer_idx}_ssm_state"
    _microscope_capture[key] = ssm_state.detach().cpu().clone()


def _resolve_ssm_slot(state_idx, layer_meta) -> int:
    if state_idx is None:
        return 0
    if state_idx.dim() == 2:
        if state_idx.shape[0] == 0:
            return 0
        last_block = getattr(layer_meta, "block_idx_last_scheduled_token", None)
        if last_block is not None and last_block.numel() > 0:
            return state_idx[0].gather(0, last_block[0:1].long()).item()
        return state_idx[0, -1].item()
    if state_idx.numel() == 0:
        return 0
    return state_idx.reshape(-1)[0].item()


def _is_prefill(layer_meta, num_tokens: int) -> bool:
    if layer_meta is not None:
        return getattr(layer_meta, "num_prefills", 0) > 0
    return num_tokens > 1


def _apply_attention_patch() -> None:
    global _attn_patched
    if _attn_patched:
        return

    try:
        from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
    except ImportError:
        try:
            from vllm.model_executor.models.qwen3_5 import Qwen3NextAttention
        except ImportError:
            logger.warning("Qwen3NextAttention not found — microscope attn patch skipped")
            return

    try:
        from vllm.forward_context import get_forward_context
    except ImportError:
        get_forward_context = None

    _original_forward = Qwen3NextAttention.forward

    def _microscope_forward(self, positions, output, hidden_states):
        global _microscope_attn_idx
        attn_layer_num = _microscope_attn_idx
        _microscope_attn_idx += 1
        layer_idx = (
            FULL_ATTN_LAYERS[attn_layer_num]
            if attn_layer_num < len(FULL_ATTN_LAYERS) else -1
        )

        is_prefill = hidden_states.shape[0] > 1
        if get_forward_context is not None:
            try:
                fwd_ctx = get_forward_context()
                attn_meta = fwd_ctx.attn_metadata
                if attn_meta is not None and isinstance(attn_meta, dict):
                    prefix = getattr(self.attn, "prefix", None)
                    if prefix:
                        layer_meta = attn_meta.get(prefix)
                        is_prefill = _is_prefill(layer_meta, hidden_states.shape[0])
            except Exception:
                pass

        if _microscope_enabled and is_prefill and layer_idx >= 0:
            microscope_capture_hidden(layer_idx, hidden_states, stage="attn_input")

        _original_forward(self, positions, output, hidden_states)

        if _microscope_enabled and is_prefill and layer_idx >= 0:
            microscope_capture_hidden(layer_idx, output, stage="attn_output")

    Qwen3NextAttention.forward = _microscope_forward
    _attn_patched = True
    logger.info("PRI microscope attention patch applied")


def _apply_deltanet_patch() -> None:
    global _deltanet_patched
    if _deltanet_patched:
        return

    try:
        from vllm.model_executor.models.qwen3_next import Qwen3NextGatedDeltaNet
    except ImportError:
        try:
            from vllm.model_executor.models.qwen3_5 import Qwen3NextGatedDeltaNet
        except ImportError:
            logger.warning("Qwen3NextGatedDeltaNet not found — microscope DN patch skipped")
            return

    from vllm.forward_context import get_forward_context

    _original_forward_core = Qwen3NextGatedDeltaNet._forward_core

    def _microscope_forward_core(self, mixed_qkv, b, a, core_attn_out):
        layer_idx = getattr(self, "layer_idx", -1)

        fwd_ctx = get_forward_context()
        attn_metadata = fwd_ctx.attn_metadata
        is_prefill = False
        layer_meta = None
        if attn_metadata is not None and isinstance(attn_metadata, dict):
            layer_meta = attn_metadata.get(self.prefix)
            if layer_meta is not None:
                is_prefill = getattr(layer_meta, "num_prefills", 0) > 0

        _original_forward_core(self, mixed_qkv, b, a, core_attn_out)

        if _microscope_enabled and is_prefill and layer_meta is not None:
            try:
                self_kv_cache = self.kv_cache[getattr(fwd_ctx, "virtual_engine", 0)]
                ssm_state = self_kv_cache[1]
                state_idx = layer_meta.non_spec_state_indices_tensor
                slot = _resolve_ssm_slot(state_idx, layer_meta)
                microscope_capture_deltanet(layer_idx, ssm_state[slot])
                microscope_capture_hidden(layer_idx, core_attn_out, stage="deltanet_out")
            except Exception:
                logger.debug("microscope deltanet capture failed L%d", layer_idx, exc_info=True)

    Qwen3NextGatedDeltaNet._forward_core = _microscope_forward_core
    _deltanet_patched = True
    logger.info("PRI microscope DeltaNet patch applied")


_apply_attention_patch()
_apply_deltanet_patch()


def _merge_extra_args(extra_args: dict) -> None:
    """Hoist nested kv_transfer_params to top-level extra_args."""
    kvtp = extra_args.get("kv_transfer_params")
    if isinstance(kvtp, dict):
        for key, value in kvtp.items():
            extra_args.setdefault(key, value)


def _apply_microscope_from_extra_args(extra_args: dict) -> None:
    microscope_on = extra_args.get("microscope")
    if microscope_on is None:
        return
    if str(microscope_on) == "off":
        disable_microscope()
        return
    tag = str(extra_args.get("microscope_tag", "capture"))
    save_path = (
        str(microscope_on)
        if str(microscope_on) not in ("1", "true", "yes")
        else "/tmp/nls_microscope"
    )
    enable_microscope(save_path=save_path, tag=tag)

    microscope_flush = extra_args.get("microscope_flush")
    if microscope_flush is not None:
        flush_microscope(tag=str(extra_args.get("microscope_tag", "capture")))


class PRIMicroscopeProcessor(LogitsProcessor):
    """Minimal logits processor: microscope enable/flush only (no router bias)."""

    @classmethod
    def validate_params(cls, params: SamplingParams) -> None:
        return

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        is_pin_memory: bool,
    ) -> None:
        self._device = device
        logger.info(
            "PRIMicroscopeProcessor initialized (attn_patched=%s dn_patched=%s)",
            _attn_patched,
            _deltanet_patched,
        )

    def is_argmax_invariant(self) -> bool:
        return True

    def update_state(self, batch_update: Optional[BatchUpdate]) -> None:
        global _microscope_attn_idx

        if _microscope_enabled and _microscope_capture:
            flush_microscope(tag=_microscope_tag)
            disable_microscope()

        _microscope_attn_idx = 0

        if batch_update is None:
            return

        for _index, params, _, _ in batch_update.added:
            if params and params.extra_args:
                _merge_extra_args(params.extra_args)
                _apply_microscope_from_extra_args(params.extra_args)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        return logits
