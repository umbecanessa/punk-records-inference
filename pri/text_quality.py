"""Detect garbled or degenerate model decode output.

Shared by ``pri.connector`` (capture hygiene — skip poisoned ``.nls`` writes when
``NLS_TURN_STRIP_GARBLED_DECODE=1``) and bench harnesses (turn sweep retries).

Heuristics: CJK noise in Latin context, abnormal vowel ratios, repeated tokens,
tool-call XML fragments without structure, path-like garbage strings.
"""

from __future__ import annotations

import re
from collections import Counter

_ALLOWED_NON_ASCII = "\u2014\u2013\u2018\u2019\u201c\u201d\u2026\u2022\u00b7\u00e9"

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]{2,}")
_VOWELS = set("aeiouAEIOU")
_TOOL_XML_RE = re.compile(r"^<?/?(?:tool_call|function|parameter)", re.I)
_PATH_LIKE_RE = re.compile(r"^/[\w./-]+$")
_ASSIGN_LIKE_RE = re.compile(r"^[\w.-]+=[\w./:-]*$")
_NORMAL_ID_RE = re.compile(r"^[A-Za-z][\w.-]*$")


def _vowel_ratio(word: str) -> float:
    if not word:
        return 0.0
    return sum(1 for c in word if c in _VOWELS) / len(word)


def _word_is_suspicious(word: str) -> bool:
    if not word:
        return True
    if _CJK_RE.search(word):
        return True
    core = re.sub(r"^[*_`#(\[\"']+|[)*_`#,\.!?:;\"']+$", "", word)
    if not core:
        return False
    if _TOOL_XML_RE.search(core):
        return True
    if re.match(r"^DECISION:?$", core, re.I):
        return False
    if _PATH_LIKE_RE.match(core):
        return False
    if _ASSIGN_LIKE_RE.match(core):
        return False
    if core.isdigit() and len(core) <= 6:
        return False
    if core in ("=", "|", "/"):
        return False
    # Paths and assignments use = / |; flag only degenerate markup chars.
    if re.search(r"[\(\)\[\]{}*#@$%^&+<>]", core):
        return True
    if len(core) >= 14 and core.isalpha() and core.islower() and _vowel_ratio(core) < 0.22:
        return True
    if len(core) >= 18 and core.isalpha() and _vowel_ratio(core) < 0.28:
        return True
    return False


def is_garbled_response(text: str) -> bool:
    """Return True when assistant text looks like model degeneration."""
    if not text or len(text.strip()) < 5:
        return True

    stripped = text.strip()
    low = stripped.lower()
    if "<tool_call>" in low or "<function=" in low or "<parameter=" in low:
        return True

    non_ascii = sum(
        1 for c in stripped
        if ord(c) > 127 and c not in _ALLOWED_NON_ASCII
    )
    if non_ascii / len(stripped) > 0.12:
        return True

    cjk = len(_CJK_RE.findall(stripped))
    latin_words = _LATIN_WORD_RE.findall(stripped)
    if cjk >= 4 and latin_words and cjk / max(len(stripped), 1) > 0.03:
        return True

    if re.search(r"(.)\1{10,}", stripped):
        return True

    words = stripped.split()
    if len(words) < 4:
        return False

    suspicious = sum(1 for w in words if _word_is_suspicious(w))
    if len(words) >= 12 and suspicious / len(words) >= 0.28:
        return True

    dup_pairs = sum(1 for i in range(len(words) - 1) if words[i] == words[i + 1])
    if dup_pairs >= 3:
        return True

    unique_ratio = len(set(words)) / len(words)
    if len(words) >= 12 and unique_ratio < 0.35:
        return True

    counts = Counter(words)
    top = counts.most_common(1)[0][1]
    if len(words) >= 10 and top / len(words) > 0.35:
        return True

    if len(words) > 20:
        for n in range(3, 7):
            for i in range(len(words) - n * 3):
                phrase = " ".join(words[i : i + n])
                if stripped.count(phrase) >= 4:
                    return True

    long_tokens = [w for w in words if len(w) >= 14]
    if long_tokens:
        messy = 0
        checked = 0
        for tok in long_tokens:
            core = re.sub(r"^[*_`#(\[\"']+|[)*_`#,\.!?:;\"']+$", "", tok)
            if not core or _NORMAL_ID_RE.match(core) or _PATH_LIKE_RE.match(core):
                continue
            checked += 1
            transitions = sum(
                1 for a, b in zip(core, core[1:])
                if a.isalpha() != b.isalpha()
            )
            if transitions >= 4:
                messy += 1
        if checked > 0 and messy / checked >= 0.4:
            return True

    return False
