from ctm_ai_eval.rag.datamodels import RagChunk


def validate_chunk_pos(doc: str, chunk: RagChunk):
    expected_text = doc[chunk.start : chunk.end]
    assert expected_text == chunk.text, (
        f"chunk {chunk.text!r} has position referring to {expected_text!r}"
    )
