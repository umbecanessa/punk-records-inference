"""Shared recall scoring for tier-1 benchmarks."""

from __future__ import annotations

import re

ALIAS_GROUPS: dict[str, list[str]] = {
    "marco": ["marco"],
    "milan": ["milan"],
    "luna": ["luna"],
    "golden retriever": ["golden retriever", "golden-retriever", "retriever"],
    "lake como": ["lake como", "como"],
    "sofia": ["sofia", "wife"],
    "hotel bellagio": ["hotel bellagio", "bellagio"],
    "bellagio": ["bellagio", "hotel bellagio"],
    "architect": ["architect", "architecture"],
}


def normalize_recall_text(text: str) -> str:
    if "<think>" in text.lower():
        text = re.sub(
            r"<think>.*?</think>",
            " ",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = re.sub(r"[^\w\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def keyword_hit(text: str, keyword: str) -> bool:
    norm = normalize_recall_text(text)
    key = keyword.lower()
    aliases = ALIAS_GROUPS.get(key, [key])
    return any(alias in norm for alias in aliases)


def score_recall_any(text: str, expected_keywords: list[str]) -> dict:
    hits = [kw for kw in expected_keywords if keyword_hit(text, kw)]
    misses = [kw for kw in expected_keywords if kw not in hits]
    return {"hits": hits, "misses": misses, "pass": len(hits) > 0}
