"""BEIR dataset integration for the RAG research pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BEIR_DATA_ROOT = PROJECT_ROOT / "data" / "beir"
BEIR_INDEX_ROOT = PROJECT_ROOT / "data" / "beir_indexes"
BEIR_RESULTS_ROOT = (
    PROJECT_ROOT / "experiments" / "Results" / "Test Runs and Ablations"
)

DATASETS = {
    "nfcorpus": {
        "beir_name": "nfcorpus",
        "display": "NFCorpus",
        "result_folder": "BEIR - NFCorpus",
    },
    "scifact": {
        "beir_name": "scifact",
        "display": "SciFact",
        "result_folder": "BEIR - SciFact",
    },
    "fiqa": {
        "beir_name": "fiqa",
        "display": "FiQA-2018",
        "result_folder": "BEIR - FiQA",
    },
}


def normalize_dataset_name(name):
    key = name.lower().strip()
    if key not in DATASETS:
        valid = ", ".join(sorted(DATASETS))
        raise KeyError(f"Unknown BEIR dataset '{name}'. Valid: {valid}")
    return key


def dataset_display_name(name):
    return DATASETS[normalize_dataset_name(name)]["display"]


def dataset_result_folder(name):
    return DATASETS[normalize_dataset_name(name)]["result_folder"]


def dataset_data_path(name):
    key = normalize_dataset_name(name)
    return BEIR_DATA_ROOT / DATASETS[key]["beir_name"]


def dataset_index_path(name):
    key = normalize_dataset_name(name)
    return BEIR_INDEX_ROOT / key


def dataset_eval_path(name):
    return dataset_data_path(name) / "eval_questions.jsonl"


def dataset_result_path(name):
    return BEIR_RESULTS_ROOT / dataset_result_folder(name)


def iter_datasets():
    for key in DATASETS:
        yield key


__all__ = [
    "BEIR_DATA_ROOT",
    "BEIR_INDEX_ROOT",
    "BEIR_RESULTS_ROOT",
    "DATASETS",
    "dataset_data_path",
    "dataset_display_name",
    "dataset_eval_path",
    "dataset_index_path",
    "dataset_result_folder",
    "dataset_result_path",
    "iter_datasets",
    "normalize_dataset_name",
]
