from dataclasses import asdict
from pathlib import Path

import polars as pl

from ctm_ai_eval.io_util import load_all_md
from ctm_ai_eval.rag.config import load_experiment_config
from ctm_ai_eval.rag.datamodels import SpanNeedle
from ctm_ai_eval.rag.needle_extraction import SpanRephraser, sample_span_needles_verbatim
from ctm_ai_eval.rag.text_processing import tokenize_words
from ctm_ai_eval.utils.hashing import stable_hash

CONFIG = load_experiment_config()
print(CONFIG)

PROMPT_DIR = Path("./assets/prompts")


def _main():
    # store everything here:
    fingerprint = stable_hash(CONFIG.model_dump(mode="json"), length=8)
    OUTDIR = Path(f"./tmp/needles/{fingerprint}")
    OUTDIR.mkdir(exist_ok=True, parents=True)

    docs = load_all_md(CONFIG.dataset.corpus_path)

    print(f"loaded {len(docs)} docs")
    assert docs

    needles_verbatim = sample_span_needles_verbatim(
        docs,
        tokenize_words,
        max_count=CONFIG.dataset.max_needles,
    )
    print(f"{len(needles_verbatim)} verbatim needles")

    pl.DataFrame([asdict(r) for r in needles_verbatim]).write_ndjson(OUTDIR / "verbatim.ndjson")

    # LLM-needles derived from verbatim needles.
    for mode in ["query", "paraphrase"]:
        print(f"generating needles ({mode})")
        prompt_sys = (PROMPT_DIR / f"needle_{mode}_sys.txt").read_text()
        prompt_user = (PROMPT_DIR / f"needle_{mode}.jinja").read_text()

        model = SpanRephraser(prompt_sys, prompt_user, CONFIG.needle_llm)
        results: list[SpanNeedle] = []
        for n in model.sample_all(needles_verbatim):
            print(n)
            results.append(n)

        pl.DataFrame([asdict(r) for r in results]).write_ndjson(OUTDIR / f"{mode}.ndjson")

    Path(OUTDIR / "cfg.json").write_text(CONFIG.model_dump_json(indent=1))

    print(f"done 🎉 (saved in {OUTDIR})")


if __name__ == "__main__":
    _main()
