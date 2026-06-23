"""Compatibility shim — vLLM CLI still references nls_vllm_plugin.* during migration."""

from pri import __version__

__all__ = ["__version__"]
