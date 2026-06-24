"""Ablation definitions — every condition overrides exactly one BASELINE key."""

from dataclasses import dataclass, field

BASELINE = {
    "chunk_size": 256,
    "overlap": 50,
    "top_k": 5,
    "retriever": "hybrid",
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "reranker": "bge",
    "query_transform": "none",
    "prompt": "strict_context_json",
    "context_filter": "none",
    "generator": "qwen2.5:14b",
    "judge": "qwen2.5:14b",
}

# Reference score from locked baseline run 007 (for delta_vs_baseline in summaries).
BASELINE_REFERENCE_SCORE = 83.22

BASELINE_FOLDER = "Locked Baseline"

DEFAULT_RUNS_PER_CONDITION = 3

# Maps BASELINE dict keys to ExperimentConfig field names where they differ.
_CONFIG_KEY_MAP = {
    "overlap": "chunk_overlap",
}


@dataclass
class AblationCondition:
    name: str
    overrides: dict = field(default_factory=dict)


@dataclass
class Ablation:
    id: str
    folder: str
    question: str
    conditions: list


ABLATIONS = [
    Ablation(
        id="A1",
        folder="Ablation 1 - Retriever Type",
        question="How much does hybrid retrieval contribute vs dense alone?",
        conditions=[
            AblationCondition("hybrid", {"retriever": "hybrid"}),
            AblationCondition("dense", {"retriever": "dense"}),
        ],
    ),
    Ablation(
        id="A2",
        folder="Ablation 2 - Generator Model Size",
        question="How much of the gain is the model vs the pipeline?",
        conditions=[
            AblationCondition("qwen2.5-14b", {"generator": "qwen2.5:14b"}),
            AblationCondition("llama3.1-8b", {"generator": "llama3.1:8b"}),
        ],
    ),
    Ablation(
        id="A3",
        folder="Ablation 3 - Output Format",
        question="Does JSON enforcement help vs XML scratchpad?",
        conditions=[
            AblationCondition("json", {"prompt": "strict_context_json"}),
            AblationCondition("xml", {"prompt": "strict_context_with_citations"}),
        ],
    ),
    Ablation(
        id="A4",
        folder="Ablation 4 - Context Filter",
        question="Is there a filter threshold that helps, or does all filtering hurt?",
        conditions=[
            AblationCondition("none", {"context_filter": "none"}),
            AblationCondition("top_sentences_10", {"context_filter": "top_sentences_10"}),
            AblationCondition("top_sentences_5", {"context_filter": "top_sentences_5"}),
        ],
    ),
    Ablation(
        id="A5",
        folder="Ablation 5 - Embedding Model",
        question="Does a larger embedding model improve retrieval measurably?",
        conditions=[
            AblationCondition("bge-small", {"embedding_model": "BAAI/bge-small-en-v1.5"}),
            AblationCondition("bge-large", {"embedding_model": "BAAI/bge-large-en-v1.5"}),
        ],
    ),
    Ablation(
        id="A6",
        folder="Ablation 6 - Reranker",
        question="How much does the reranker contribute?",
        conditions=[
            AblationCondition("bge", {"reranker": "bge"}),
            AblationCondition("none", {"reranker": "none"}),
        ],
    ),
    Ablation(
        id="A7",
        folder="Ablation 7 - Query Transformation",
        question="Does HyDE help on abstractly phrased questions?",
        conditions=[
            AblationCondition("none", {"query_transform": "none"}),
            AblationCondition("hyde", {"query_transform": "hyde"}),
        ],
    ),
    Ablation(
        id="A8",
        folder="Ablation 8 - Top-K",
        question="Does giving the reranker more candidates help?",
        conditions=[
            AblationCondition("top5", {"top_k": 5}),
            AblationCondition("top7", {"top_k": 7}),
            AblationCondition("top10", {"top_k": 10}),
        ],
    ),
    Ablation(
        id="A9",
        folder="Ablation 9 - Chunk Size",
        question="Is 256 tokens actually optimal?",
        conditions=[
            AblationCondition("128", {"chunk_size": 128}),
            AblationCondition("256", {"chunk_size": 256}),
            AblationCondition("512", {"chunk_size": 512}),
        ],
    ),
]


def get_ablation(ablation_id):
    key = ablation_id.upper()
    if not key.startswith("A"):
        key = f"A{ablation_id}"
    for ablation in ABLATIONS:
        if ablation.id == key:
            return ablation
    raise KeyError(f"Unknown ablation ID: {ablation_id}")


def merged_settings(overrides=None):
    settings = dict(BASELINE)
    if overrides:
        settings.update(overrides)
    return settings


def to_experiment_config(settings, name, round_name, description=""):
    """Build ExperimentConfig from merged BASELINE settings."""
    from experiment_config import ExperimentConfig

    kwargs = {}
    for key, value in settings.items():
        field_name = _CONFIG_KEY_MAP.get(key, key)
        kwargs[field_name] = value

    return ExperimentConfig(
        name=name,
        round=round_name,
        description=description,
        **kwargs,
    )
