from ctm_ai_eval import config
from ctm_ai_eval.io_util import load_all_md
from ctm_ai_eval.rag.needle_extraction import sample_span_needles_verbatim
from ctm_ai_eval.rag.text_processing import tokenize_words

CONFIG = config.load_config()
print(CONFIG)

docs = load_all_md(CONFIG.dataset.corpus_path)

print(f"loaded {len(docs)} docs")
assert docs

needles = sample_span_needles_verbatim(
    docs,
    tokenize_words,
    max_count=CONFIG.dataset.max_needles,
)

for n in needles:
    print("-" * 80 + "\n", n.query)
