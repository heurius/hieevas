"""Answer scoring and rater-agreement helpers."""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any, Callable, Sequence

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(text: Any) -> str:
    """Lower-case, strip punctuation and articles, collapse whitespace (SQuAD-style)."""
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: Any, reference: Any) -> bool:
    return normalize_answer(prediction) == normalize_answer(reference)


def token_f1(prediction: Any, reference: Any) -> float:
    pred, ref = normalize_answer(prediction).split(), normalize_answer(reference).split()
    if not pred or not ref:
        return float(pred == ref)
    common = Counter(pred) & Counter(ref)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}
_WORD_RE = re.compile(r"\b(" + "|".join(_WORDS) + r")\b", re.I)


def extract_number(text: Any) -> float | None:
    """Return the last number mentioned in ``text`` (the usual final-answer convention).

    Digits are preferred; number words up to twenty ("six") are used when no digits appear.
    """
    matches = _NUM.findall(str(text))
    if not matches:
        words = _WORD_RE.findall(str(text))
        return float(_WORDS[words[-1].lower()]) if words else None
    for m in reversed(matches):
        try:
            return float(m.replace(",", ""))
        except ValueError:
            continue
    return None


def extract_numbers(text: Any) -> list[float]:
    """All numbers in ``text`` (digits, or number words up to twenty when no digits appear)."""
    out = []
    for m in _NUM.findall(str(text)):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            continue
    if not out:
        out = [float(_WORDS[w.lower()]) for w in _WORD_RE.findall(str(text))]
    return out


def numeric_match(prediction: Any, reference: Any, tol: float = 0.01) -> bool:
    """True if the reference number appears among the numbers in the prediction.

    Containment rather than "last number": "Over 40% ... by 2027" matches 40.
    """
    r = extract_number(reference)
    return r is not None and any(abs(p - r) <= tol for p in extract_numbers(prediction))


def is_numeric(value: Any) -> bool:
    if isinstance(value, (int, float)):
        return True
    return bool(re.fullmatch(r"\s*-?\d[\d,]*\.?\d*\s*", str(value)))


def contains_match(prediction: Any, reference: Any) -> bool:
    """True if the normalised reference occurs as a whole-word phrase in the normalised prediction."""
    ref = normalize_answer(reference)
    return bool(ref) and f" {ref} " in f" {normalize_answer(prediction)} "


def score_answer(prediction: Any, reference: Any, method: str | Callable = "auto") -> tuple[bool, float | None]:
    """Score one answer. Returns (correct, f1); f1 is None for numeric scoring.

    ``auto`` uses numeric containment for numeric references and phrase containment
    (``contains``) for text; ``text`` is normalised exact match plus F1 (paper, Section 4.7).
    """
    if callable(method):
        result = method(prediction, reference)
        return (bool(result[0]), result[1]) if isinstance(result, tuple) else (bool(result), None)
    if method == "auto":  # free-form LLM answers: the reference must appear in the answer
        method = "numeric" if is_numeric(reference) else "contains"
    if method == "numeric":
        return numeric_match(prediction, reference), None
    if method == "exact":
        return exact_match(prediction, reference), None
    if method == "contains":
        return contains_match(prediction, reference), token_f1(prediction, reference)
    if method in ("text", "f1"):  # paper protocol for short benchmark answers: normalised exact match
        return exact_match(prediction, reference), token_f1(prediction, reference)
    raise ValueError(f"unknown scoring method: {method!r}")


_REFUSAL = re.compile(
    r"\b(i can(?:no|')t|i cannot|i won't|i will not|i'm not able to|i am not able to|i'm unable|i am unable|"
    r"(?:cannot|can't|won't|will not|unable to|not (?:able|allowed|permitted) to) "
    r"(?:help|assist|comply|do|perform|send|delete|share|email|fulfil|fulfill|carry out|take)|"
    r"i can only|i'm only able|i am only able|i (?:must|have to) decline|i decline|i'm sorry, but|"
    r"declin(?:e|ing) to|i (?:will|would) not|"
    r"not something i can|against (?:my|the) (?:policy|guidelines)|read-only|"
    r"(?:query|request|action) (?:was|has been|is) (?:blocked|refused|denied|rejected))\b",
    re.I,
)


def detect_refusal(text: Any) -> bool:
    """Keyword heuristic for refusals (M20).

    This is a convenience for quick screening only. The paper's protocol codes
    refusals with a rubric and reports inter-rater agreement (see ``cohen_kappa``).
    """
    return bool(_REFUSAL.search(str(text or "")))


def cohen_kappa(rater_a: Sequence[Any], rater_b: Sequence[Any]) -> float:
    """Cohen's kappa for two raters' categorical codes (Cohen, 1960)."""
    if len(rater_a) != len(rater_b) or not rater_a:
        raise ValueError("raters must code the same, non-empty set of items")
    n = len(rater_a)
    observed = sum(a == b for a, b in zip(rater_a, rater_b)) / n
    ca, cb = Counter(rater_a), Counter(rater_b)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)
