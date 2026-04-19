from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExperimentPaths:
    setup_hash: str
    root: Path = field(default=Path("./tmp"))

    def available_setups(self):
        return sorted([p.stem for p in self.root.glob("rag_haystack/*/")])

    @property
    def _haystack_dir(self) -> Path:
        d = self.root / "rag_haystack" / self.setup_hash

        d.mkdir(exist_ok=True, parents=True)
        return d

    def _haystack_target_dir(self, target_id: str) -> Path:
        d = self._haystack_dir / target_id

        d.mkdir(exist_ok=True, parents=True)
        return d

    @property
    def haystack_chunks(self):
        return self._haystack_dir / "chunks.parquet"

    @property
    def haystack_needles(self):
        return self._haystack_dir / "needles.parquet"

    def haystack_target_cfg(self, target_id: str):
        return self._haystack_target_dir(target_id) / "cfg.json"

    def haystack_target_res(self, target: str):
        return self._haystack_target_dir(target) / "result.parquet"

    def available_targets(self):
        return sorted([p.parent.stem for p in self._haystack_dir.rglob("result.parquet")])
