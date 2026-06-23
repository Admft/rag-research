import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path

from config import OLLAMA_JUDGE_MODEL, PROJECT_ROOT

GRID_FILE = PROJECT_ROOT / "experiments" / "grid.json"


@dataclass
class ExperimentConfig:
    name: str
    chunk_size: int = 512
    chunk_overlap: int = 50
    retriever: str = "dense"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    top_k: int = 5
    reranker: str = "none"
    query_transform: str = "none"
    prompt: str = "strict_context"
    context_filter: str = "none"
    generator: str = "qwen2.5:14b"
    judge: str = OLLAMA_JUDGE_MODEL
    round: str = "baseline"
    description: str = ""

    def index_key(self):
        return (self.chunk_size, self.chunk_overlap, self.embedding_model)

    def to_dict(self):
        return asdict(self)


def load_grid():
    with GRID_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_experiment_configs(grid=None):
    grid = grid or load_grid()
    baseline = deepcopy(grid["baseline"])
    runs = []

    for entry in grid["runs"]:
        config = deepcopy(baseline)
        config.update({k: v for k, v in entry.items() if k != "overrides"})
        overrides = entry.get("overrides", {})
        config.update(overrides)
        runs.append(ExperimentConfig(**config))

    return runs


def get_run_by_name(name):
    for config in build_experiment_configs():
        if config.name == name:
            return config
    raise KeyError(f"Unknown experiment run: {name}")


def get_runs_for_round(round_name):
    return [cfg for cfg in build_experiment_configs() if cfg.round == round_name]
