import sys
from pathlib import Path
from typing import Literal

from ctm_ai_eval.rag import text_processing
from ctm_ai_eval.rag.chunkers.basic_chunking import TokenChunker
from ctm_ai_eval.rag.needle_extraction import sample_needles_llm
from ctm_ai_eval.utils.io_util import load_all_md


def _llm_needle_example():
    print("Lets try getting some needles...")

    corpus_path = Path(sys.argv[1])
    assert corpus_path.exists(), f"could not find {corpus_path}"

    # prepare data
    docs = load_all_md(corpus_path)
    chunks = TokenChunker(100, text_processing.tokenize_words)(docs)

    PROMPT_DIR = Path("./ai_eval/prompts")
    MODE: Literal["paraphrase", "query"] = "query"

    sample_needles_llm(chunks, PROMPT_DIR, MODE, verbose=True, max_count=7)


if __name__ == "__main__":
    _llm_needle_example()
