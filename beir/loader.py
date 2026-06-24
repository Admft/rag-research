"""Load BEIR corpus, queries, and qrels."""

import json

from . import dataset_data_path, dataset_eval_path, normalize_dataset_name
from .beir_lib import load_generic_data_loader, require_beir_library


def load_beir_split(name, split="test"):
    require_beir_library()
    GenericDataLoader = load_generic_data_loader()

    key = normalize_dataset_name(name)
    data_path = dataset_data_path(key)
    if not data_path.exists():
        raise FileNotFoundError(
            f"BEIR dataset '{key}' is not downloaded at {data_path}.\n"
            f"Run: python3 run_beir.py --download"
        )

    corpus, queries, qrels = GenericDataLoader(str(data_path)).load(split=split)
    return corpus, queries, qrels


def document_text(doc):
    title = (doc.get("title") or "").strip()
    text = (doc.get("text") or "").strip()
    if title and text:
        return f"{title}\n\n{text}"
    return title or text


def corpus_to_documents(corpus):
    documents = []
    for doc_id, doc in corpus.items():
        text = document_text(doc)
        if not text:
            continue
        documents.append({
            "source": str(doc_id),
            "format": "beir",
            "text": text,
        })
    return documents


def load_eval_questions(name):
    path = dataset_eval_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"Converted eval questions not found at {path}.\n"
            f"Run: python3 run_beir.py --index  (builds eval_questions.jsonl)"
        )

    questions = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions
