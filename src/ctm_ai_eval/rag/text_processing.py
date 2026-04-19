import re
from difflib import SequenceMatcher

from ctm_ai_eval.rag.datamodels import SpanToken


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\W+", " ", text)
    return text.strip()


def tokenize_words(text: str) -> list[SpanToken]:
    """Return (token, start, end)."""
    return [SpanToken(m.group(), m.start(), m.end()) for m in re.finditer(r"\w+", text)]


def tokenize_sentences(text: str) -> list[SpanToken]:
    spans: list[SpanToken] = []
    for m in re.finditer(r"[^.!?\n]+[.!?]?", text):
        s = m.group().strip()
        if s:
            spans.append(SpanToken(s, m.start(), m.end()))
    return spans


def tokenize_sentence_windows(text: str, window_size: int = 3):
    sents = tokenize_sentences(text)
    out: list[SpanToken] = []

    for i in range(len(sents)):
        window = sents[i : i + window_size]
        if not window:
            continue

        start = window[0].start
        end = window[-1].end
        span_text = text[start:end]

        out.append(SpanToken(span_text, start, end))

    return out


def longest_common_substring(a: str, b: str) -> int:
    """Length of longest common substring (DP, O(n*m), fine for baseline)."""
    if not a or not b:
        return 0

    dp = [0] * (len(b) + 1)
    max_len = 0

    for i in range(1, len(a) + 1):
        new_dp = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                new_dp[j] = dp[j - 1] + 1
                max_len = max(max_len, new_dp[j])
        dp = new_dp

    return max_len


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()
