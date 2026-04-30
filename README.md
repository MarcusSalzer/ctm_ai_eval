# CTM AI Eval

A framework for evaluating retrieval and chat APIs. It is quite experimental and inconvenient. 

It can run two types of evaluation:

- Retrieval "Haystack"
- End-to-end "QA"

All entrypoints are in `scripts/`. 

Some parts use a TOML-based config, some parts use options inlined in the source, maybe i should fix that.


## Retrieval "Haystack"

Get a sense of how well retrieval works on a corpus, needs no additional data.

- Generate "needles" from the corpus: `prepare_needles.py`.
    - Three difficulties: verbatim/paraphrase/query.
- Then run experiments that evaluate `HaystackTargets` on all needles.
    - Each target specifies: document loader, chunker, retriever

## End-to-end "QA"

Needs a QA-dataset. Currently a small dataset is available in `assets/data`. Optionally needs a corpus for RAG.

- setp 1: Trace Chat-API on a curated dataset. 
    - The targets specify model, temperature, system_prompt and can optionally have RAG.
- step 2: Compute metrics
    - Quality scores in [0, 1] (LlmJudge & HumanJudge CLI), Heuristics (IsConscise)

- Current limitation: there is no rag-friendly prompt engineering. 

## Future improvements?

- Try other document loaders than `load_all_md()` (e.g. PDF.)
- If time allows, one should clean up the configurations etc.
- Adapter for using a larger dataset (e.g. CS1QA)
- Prompt optimization: It would be interesting to for instance try the GEPA-approach.
- Hyperparamter tuning: there are many parameters in the whole pipline. Tuning one or two at a time with optuna would be interesting.