"""NLS Memory Management V2 — Admin API middleware for vLLM.

Intercepts /admin/memory/* requests and handles them directly,
passing all other requests through to the vLLM server unchanged.

Usage in docker/start.sh:
    --middleware pri.admin.NLSAdminMiddleware

Endpoints:
    GET  /admin/memory/stats           — memory store statistics
    POST /admin/memory/reload          — hot-reload index + BM25 from disk
    POST /admin/memory/warmup          — JL #20.5e: prime cold-start caches
                                          (sentence-transformers model load,
                                          delta_fp/ffn_sig cache verification,
                                          synthetic encode pass)
    POST /admin/memory/delete          — delete memories by session_id
    GET  /admin/memory/search          — run Swiss-Cheese retrieval (read-only debug)
    GET  /admin/memory/user-memories   — list memories for a specific user
    GET  /admin/memory/token-stats     — per-user token counts and savings estimate
    GET  /admin/memory/retrieval-log   — retrieval events for a specific request
"""

from __future__ import annotations

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger("nls_admin_api")


_retrieval_log: dict[str, dict] = {}
_RETRIEVAL_LOG_MAX = 200
_RETRIEVAL_LOG_PATH: str | None = None


def _get_retrieval_log_path() -> str:
    global _RETRIEVAL_LOG_PATH
    if _RETRIEVAL_LOG_PATH is None:
        store = _get_store()
        if store is not None:
            _RETRIEVAL_LOG_PATH = str(store._dir / "retrieval_log.jsonl")
        else:
            _RETRIEVAL_LOG_PATH = "/workspace/kv_snapshots/retrieval_log.jsonl"
    return _RETRIEVAL_LOG_PATH


def record_retrieval_event(request_id: str, event: dict):
    """Record a retrieval event to shared disk file (called from EngineCore process)."""
    import json as _json
    _retrieval_log[request_id] = event
    if len(_retrieval_log) > _RETRIEVAL_LOG_MAX:
        oldest = next(iter(_retrieval_log))
        del _retrieval_log[oldest]
    try:
        path = _get_retrieval_log_path()
        entry = {"request_id": request_id, "timestamp": time.time(), **event}
        with open(path, "a") as f:
            f.write(_json.dumps(entry, separators=(",", ":"), default=str) + "\n")
    except Exception:
        pass


def _read_retrieval_log_from_disk(user_id: str = "", limit: int = 20) -> list[dict]:
    """Read recent retrieval events from shared disk file (called from APIServer process)."""
    import json as _json
    path = _get_retrieval_log_path()
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        events = []
        for line in reversed(lines[-200:]):
            line = line.strip()
            if not line:
                continue
            evt = _json.loads(line)
            if user_id and evt.get("user_id", "") != user_id:
                continue
            events.append(evt)
            if len(events) >= limit:
                break
        return events
    except FileNotFoundError:
        return []
    except Exception:
        return []


def _get_store():
    """Lazily fetch the auto_memory store singleton."""
    try:
        from pri import retrieve as auto_memory
        if auto_memory.is_enabled() and auto_memory._store is not None:
            return auto_memory._store
    except ImportError:
        pass
    return None


_last_index_check: float = 0.0
_last_index_offset: int = 0


def _maybe_reload(store) -> None:
    """Incrementally append new JSONL entries from disk (EngineCore appends, we tail)."""
    global _last_index_check, _last_index_offset
    import json as _json
    from dataclasses import fields as _fields

    now = time.time()
    if now - _last_index_check < 1:
        return
    _last_index_check = now
    try:
        idx_path = store._dir / "index.jsonl"
        sz = idx_path.stat().st_size
        if sz <= _last_index_offset:
            return

        with open(idx_path, "r", encoding="utf-8") as f:
            f.seek(_last_index_offset)
            new_lines = f.readlines()
        _last_index_offset = sz

        from pri.store import Memory
        valid_fields = {fld.name for fld in _fields(Memory)}
        added = 0
        known_ids = {m.id for m in store._memories}
        for line in new_lines:
            line = line.strip()
            if not line:
                continue
            raw = _json.loads(line)
            filtered = {k: v for k, v in raw.items() if k in valid_fields}
            if "session_id" not in filtered:
                filtered["session_id"] = ""
            mem = Memory(**filtered)
            if mem.id in known_ids:
                continue
            store._memories.append(mem)
            known_ids.add(mem.id)
            added += 1

        if added > 0:
            logger.info("Admin incremental sync: +%d memories (total %d)", added, len(store._memories))
    except Exception as e:
        logger.debug("_maybe_reload: %s", e)


class NLSAdminMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        if not path.startswith("/admin/memory"):
            return await call_next(request)

        try:
            if path == "/admin/memory/stats" and request.method == "GET":
                return self._handle_stats()

            if path == "/admin/memory/reload" and request.method == "POST":
                return self._handle_reload()

            if path == "/admin/memory/warmup" and request.method == "POST":
                try:
                    body = await request.json()
                except Exception:
                    body = {}
                return self._handle_warmup(body)

            if path == "/admin/memory/delete" and request.method == "POST":
                body = await request.json()
                return self._handle_delete(body)

            if path == "/admin/memory/search" and request.method in ("GET", "POST"):
                if request.method == "POST":
                    body = await request.json()
                else:
                    body = dict(request.query_params)
                return self._handle_search(body)

            if path == "/admin/memory/user-memories" and request.method == "GET":
                params = dict(request.query_params)
                return self._handle_user_memories(params)

            if path == "/admin/memory/token-stats" and request.method == "GET":
                params = dict(request.query_params)
                return self._handle_token_stats(params)

            if path == "/admin/memory/retrieval-log" and request.method == "GET":
                params = dict(request.query_params)
                return self._handle_retrieval_log(params)

            return JSONResponse(
                {"error": f"Unknown admin endpoint: {path}"},
                status_code=404,
            )
        except Exception as e:
            logger.error("Admin API error on %s: %s", path, e, exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    def _handle_stats(self) -> JSONResponse:
        store = _get_store()
        if store is None:
            return JSONResponse(
                {"error": "Memory store not initialized"}, status_code=503,
            )
        stats = store.get_stats()
        return JSONResponse(stats)

    def _handle_reload(self) -> JSONResponse:
        store = _get_store()
        if store is None:
            return JSONResponse(
                {"error": "Memory store not initialized"}, status_code=503,
            )
        t0 = time.perf_counter()
        result = store.reload()
        result["elapsed_ms"] = round((time.perf_counter() - t0) * 1000)
        logger.info("Admin reload completed: %s", result)
        return JSONResponse(result)

    def _handle_warmup(self, body: dict) -> JSONResponse:
        """JL #20.5e Step 2 — prime cold-start caches.

        Triggers eager initialization of the components that
        otherwise lazy-load on the first chat request and add a
        one-shot warmup tax (most visibly: the sentence-transformers
        model load, ~5–10s of TTFT on F01). Calling this endpoint
        once after container startup absorbs that tax out of band so
        subsequent inferences hit warm paths from the very first
        request.

        Body (all optional):
            {
                "encode_probe": "<query string>",   # default "warmup probe"
                "skip_embedder": false,             # for diagnostics only
                "skip_search_probe": false          # default skips search
            }

        Returns per-step timings + cache occupancy stats:
            {
                "status": "ok",
                "timings_ms": { "embedder_load_ms": ..., ... },
                "details": {
                    "delta_fp": { "loaded", "shape", "recovered_count" },
                    "ffn_sig": { ... same shape ... },
                    "embedder": { "loaded", "was_already_loaded",
                                  "dim", "model", "encode_ok" }
                }
            }

        Idempotent: a second call short-circuits on the embedder
        (was_already_loaded=True) and just re-verifies the cache
        occupancy. Useful as a liveness probe + warmth indicator
        for orchestration (e.g. a healthcheck that flips
        ready=true only once delta_fp.recovered_count >= N and
        embedder.loaded == True).
        """
        store = _get_store()
        if store is None:
            return JSONResponse(
                {"error": "Memory store not initialized"}, status_code=503,
            )

        try:
            import numpy as _np
        except Exception as e:
            return JSONResponse(
                {"error": f"numpy unavailable: {e}"}, status_code=500,
            )

        timings: dict[str, float] = {}
        details: dict[str, dict] = {}

        t0 = time.perf_counter()
        delta_arr = getattr(store, "_delta_fp_cache", None)
        delta_recovered = 0
        delta_shape: list[int] | None = None
        if delta_arr is not None:
            delta_shape = list(delta_arr.shape)
            try:
                norms = _np.linalg.norm(delta_arr, axis=1)
                delta_recovered = int((norms > 1e-8).sum())
            except Exception:
                delta_recovered = -1
        timings["delta_fp_check_ms"] = round(
            (time.perf_counter() - t0) * 1000, 2,
        )
        details["delta_fp"] = {
            "loaded": delta_arr is not None,
            "shape": delta_shape,
            "recovered_count": delta_recovered,
        }

        t0 = time.perf_counter()
        ffn_arr = getattr(store, "_ffn_sig_cache", None)
        ffn_recovered = 0
        ffn_shape: list[int] | None = None
        if ffn_arr is not None:
            ffn_shape = list(ffn_arr.shape)
            try:
                norms = _np.linalg.norm(ffn_arr, axis=1)
                ffn_recovered = int((norms > 1e-6).sum())
            except Exception:
                ffn_recovered = -1
        timings["ffn_sig_check_ms"] = round(
            (time.perf_counter() - t0) * 1000, 2,
        )
        details["ffn_sig"] = {
            "loaded": ffn_arr is not None,
            "shape": ffn_shape,
            "recovered_count": ffn_recovered,
        }

        skip_embedder = bool(body.get("skip_embedder", False))
        embedder_info: dict = {
            "loaded": False,
            "was_already_loaded": False,
            "dim": 0,
            "model": None,
            "encode_ok": None,
        }
        if not skip_embedder:
            t0 = time.perf_counter()
            try:
                from pri.store import (
                    SentenceEmbedder, _SEMANTIC_MODEL_NAME,
                )
                embedder = SentenceEmbedder.get()
                was_loaded = embedder._model is not None
                ok = embedder._ensure_loaded()
                embedder_info["loaded"] = bool(ok)
                embedder_info["was_already_loaded"] = bool(was_loaded)
                embedder_info["dim"] = int(embedder._dim or 0)
                embedder_info["model"] = _SEMANTIC_MODEL_NAME
            except Exception as e:
                embedder_info["error"] = str(e)
            timings["embedder_load_ms"] = round(
                (time.perf_counter() - t0) * 1000, 2,
            )

            if embedder_info["loaded"]:
                probe_text = str(
                    body.get("encode_probe") or "warmup probe query"
                )
                t0 = time.perf_counter()
                try:
                    _ = embedder.encode_query(probe_text)
                    embedder_info["encode_ok"] = True
                except Exception as e:
                    embedder_info["encode_ok"] = False
                    embedder_info["encode_error"] = str(e)
                timings["embedder_encode_ms"] = round(
                    (time.perf_counter() - t0) * 1000, 2,
                )
        details["embedder"] = embedder_info

        timings["total_ms"] = round(
            sum(v for v in timings.values()), 2,
        )

        logger.info(
            "Admin warmup completed: total=%dms, embedder_load=%dms "
            "(was_loaded=%s), delta_fp_recovered=%d/%s, "
            "ffn_sig_recovered=%d/%s",
            timings["total_ms"],
            timings.get("embedder_load_ms", 0),
            embedder_info.get("was_already_loaded"),
            delta_recovered,
            delta_shape[0] if delta_shape else "?",
            ffn_recovered,
            ffn_shape[0] if ffn_shape else "?",
        )

        return JSONResponse({
            "status": "ok",
            "timings_ms": timings,
            "details": details,
        })

    def _handle_delete(self, body: dict) -> JSONResponse:
        store = _get_store()
        if store is None:
            return JSONResponse(
                {"error": "Memory store not initialized"}, status_code=503,
            )
        session_ids = body.get("session_ids", [])
        if not session_ids:
            return JSONResponse(
                {"error": "session_ids required"}, status_code=400,
            )
        result = store.delete_by_session_ids(session_ids)
        logger.info("Admin delete: %s", result)
        return JSONResponse(result)

    def _handle_search(self, body: dict) -> JSONResponse:
        store = _get_store()
        if store is None:
            return JSONResponse(
                {"error": "Memory store not initialized"}, status_code=503,
            )
        query = body.get("query", "")
        top_k = int(body.get("top_k", 20))
        user_id = body.get("user_id", "bench")

        if not query:
            return JSONResponse(
                {"error": "query text required"}, status_code=400,
            )

        try:
            from pri import retrieve as auto_memory
            auto_memory._load_embed_weights_lazy()
            from pri.store import compute_fingerprint

            from transformers import AutoTokenizer
            import os
            model_path = os.environ.get("NLS_MODEL_PATH") or os.environ.get(
                "MODEL_PATH", "/model"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            token_ids = tokenizer.encode(query)

            results = store.search_swiss_cheese(
                token_ids, top_k=top_k,
                user_id=user_id if user_id != "default" else None,
            )

            matches = []
            for mem, score in (results or []):
                matches.append({
                    "session_id": mem.session_id,
                    "score": round(score, 4),
                    "num_tokens": mem.num_tokens,
                    "ring_type": mem.ring_type,
                    "kv_path": mem.kv_path,
                    "user_id": mem.user_id,
                })

            return JSONResponse({
                "query": query,
                "top_k": top_k,
                "results": matches,
                "count": len(matches),
            })
        except Exception as e:
            logger.error("Search failed: %s", e, exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    def _handle_user_memories(self, params: dict) -> JSONResponse:
        store = _get_store()
        if store is None:
            return JSONResponse(
                {"error": "Memory store not initialized"}, status_code=503,
            )
        _maybe_reload(store)
        user_id = params.get("user_id", "")
        limit = int(params.get("limit", "500"))
        include_kv = params.get("include_kv", "") in ("1", "true", "yes")

        if not user_id:
            return JSONResponse(
                {"error": "user_id required"}, status_code=400,
            )

        memories = []
        for mem in store._memories:
            if mem.user_id != user_id:
                continue
            text = mem.description or ""
            if not text and hasattr(mem, 'kv_path') and mem.kv_path:
                try:
                    from pri.format import read_manifest
                    manifest = read_manifest(mem.kv_path)
                    text = manifest.get("conversation_text", "") or ""
                except Exception:
                    pass
            entry = {
                "id": mem.id,
                "sessionId": mem.session_id,
                "role": getattr(mem, "role", "user"),
                "timestamp": mem.timestamp,
                "preview": text[:120],
                "baseSessionId": getattr(mem, "base_session_id", ""),
                "turnIndex": getattr(mem, "turn_index", -1),
                "prevHash": getattr(mem, "prev_hash", ""),
            }
            if include_kv:
                entry["kvPath"] = mem.kv_path or ""
                entry["numTokens"] = mem.num_tokens
            memories.append(entry)

        if include_kv:
            memories.sort(
                key=lambda m: (m.get("turnIndex", -1), m.get("timestamp", 0)),
            )
            if limit > 0:
                memories = memories[-limit:]
        else:
            memories.sort(key=lambda m: m["timestamp"], reverse=True)
            memories = memories[:limit]

        return JSONResponse({
            "user_id": user_id,
            "memories": memories,
            "count": len(memories),
        })

    def _handle_token_stats(self, params: dict) -> JSONResponse:
        store = _get_store()
        if store is None:
            return JSONResponse(
                {"error": "Memory store not initialized"}, status_code=503,
            )
        _maybe_reload(store)
        user_id = params.get("user_id", "")
        if not user_id:
            return JSONResponse(
                {"error": "user_id required"}, status_code=400,
            )

        total_stored = 0
        user_memory_count = 0
        for mem in store._memories:
            if mem.user_id != user_id:
                continue
            total_stored += mem.num_tokens
            user_memory_count += 1

        injected = 0
        prompt_tokens = 0
        user_events = _read_retrieval_log_from_disk(user_id=user_id, limit=10)
        for evt in user_events:
            if evt.get("type") == "swap_stats":
                continue
            injected = evt.get("injected_tokens", 0) or evt.get("total_tokens", 0)
            prompt_tokens = evt.get("prompt_tokens", 0)
            break

        if user_memory_count == 0:
            injected = 0
            prompt_tokens = 0

        # Without NLS you'd send system + full history + current msg.
        # With NLS you send only system + current msg (prompt_tokens) and
        # get `injected` phantom tokens for free.
        # traditional = what the prompt would cost if history were sent instead
        traditional = prompt_tokens + injected
        saved = injected
        pct = (saved / traditional * 100) if traditional > 0 else 0

        return JSONResponse({
            "user_id": user_id,
            "memory_count": user_memory_count,
            "total_stored_tokens": total_stored,
            "injectedTokens": injected,
            "promptTokens": prompt_tokens,
            "traditionalContextTokens": traditional,
            "savedTokens": saved,
            "savingsPercent": round(pct, 1),
        })

    def _handle_retrieval_log(self, params: dict) -> JSONResponse:
        user_id = params.get("user_id", "")
        events = _read_retrieval_log_from_disk(user_id=user_id, limit=20)
        return JSONResponse({"events": events, "count": len(events)})
