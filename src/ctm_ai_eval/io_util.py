from pathlib import Path


def load_all_md(root: Path):
    """Load all .md/.qmd recursively."""
    docs = sorted(root.rglob("*.md")) + sorted(root.rglob("*.qmd"))
    print(f"{len(docs)=}")

    return [p.read_text() for p in docs]
