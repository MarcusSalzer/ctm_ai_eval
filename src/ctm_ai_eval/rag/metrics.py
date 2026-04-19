from ctm_ai_eval.rag.datamodels import RagChunk, RetrievalResult, SpanNeedle

# ==== Exact chunk metrics ===


def recall_at_k(results: list[RetrievalResult], gt_id: int, *, k: int) -> float:
    """1.0 if correct is included."""
    if k > len(results):
        raise ValueError(f"cannot eval {k=}, {len(results)=}")
    return float(any(r.chunk.id == gt_id for r in results[:k]))


def reciprocal_rank(results: list[RetrievalResult], gt_id: int) -> float:
    """1.0 if first, 1/k if at rank k."""
    for i, r in enumerate(results):
        if r.chunk.id == gt_id:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k_doc(results: list[RetrievalResult], gt_id: int, *, k: int) -> float:
    """1.0 if correct is included."""
    if k > len(results):
        raise ValueError(f"cannot eval {k=}, {len(results)=}")
    return float(any(r.chunk.doc_id == gt_id for r in results[:k]))


def reciprocal_rank_doc(results: list[RetrievalResult], gt_id: int) -> float:
    """1.0 if first, 1/k if at rank k."""
    for i, r in enumerate(results):
        if r.chunk.doc_id == gt_id:
            return 1.0 / (i + 1)
    return 0.0


# ==== Span metrics ====


def overlap(a_start: int, a_end: int, b_start: int, b_end: int):
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def overlap_ratio(span: SpanNeedle, chunk: RagChunk) -> float:
    """NOTE: normalized by needle span."""
    if span.doc_id != chunk.doc_id:
        return 0.0

    inter = overlap(
        span.start_char,
        span.end_char,
        chunk.start,
        chunk.start + len(chunk.text),
    )
    span_len = span.end_char - span.start_char
    return inter / span_len if span_len > 0 else 0.0


def recall_at_k_span(
    results: list[RetrievalResult],
    span: SpanNeedle,
    *,
    k: int,
    threshold: float = 0.0,  # 0.0 = any overlap, ->1 full overlap
) -> float:
    if k > len(results):
        raise ValueError(f"cannot eval {k=}, {len(results)=}")

    return float(any(overlap_ratio(span, r.chunk) > threshold for r in results[:k]))


def reciprocal_rank_span(
    results: list[RetrievalResult],
    span: SpanNeedle,
    *,
    threshold: float = 0.0,
) -> float:
    for i, r in enumerate(results):
        if overlap_ratio(span, r.chunk) > threshold:
            return 1.0 / (i + 1)
    return 0.0


def soft_reciprocal_rank(
    results: list[RetrievalResult],
    span: SpanNeedle,
) -> float:
    """Reciprocal rank graded by overlap ratio."""
    best = 0.0
    for i, r in enumerate(results):
        rel = overlap_ratio(span, r.chunk)
        score = rel / (i + 1)
        best = max(best, score)
    return best


def max_overlap_at_k(
    results: list[RetrievalResult],
    span: SpanNeedle,
    *,
    k: int,
) -> float:
    """What is the best overlap-ratio in the top-k"""
    assert k <= len(results)
    return max(overlap_ratio(span, r.chunk) for r in results[:k])
