from config import (
    DATASET_STAGE,
    EVAL_FILE,
    MILESTONE_NAME,
    TARGET_DOC_COUNT_MAX,
    TARGET_DOC_COUNT_MIN,
    TARGET_QUESTION_COUNT,
)
from documents import list_raw_documents


def count_raw_documents():
    return len(list_raw_documents())


def count_eval_questions():
    if not EVAL_FILE.exists():
        return 0

    count = 0
    with EVAL_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def progress_bar(current, target):
    width = 20
    ratio = min(current / target, 1.0)
    filled = int(width * ratio)
    return f"[{'#' * filled}{'-' * (width - filled)}] {current}/{target}"


def main():
    doc_count = count_raw_documents()
    question_count = count_eval_questions()

    print("=" * 72)
    print(MILESTONE_NAME)
    print("=" * 72)
    print(f"Dataset stage: {DATASET_STAGE}")
    print()

    print("Progress")
    print("-" * 40)
    print(f"Documents in data/raw: {progress_bar(doc_count, TARGET_DOC_COUNT_MIN)}")
    print(f"  Target range: {TARGET_DOC_COUNT_MIN}-{TARGET_DOC_COUNT_MAX} public .txt or .pdf files")
    print()
    print(f"Eval questions:        {progress_bar(question_count, TARGET_QUESTION_COUNT)}")
    print(f"  File: {EVAL_FILE}")
    print()

    print("Checklist")
    print("-" * 40)
    print(f"[{'x' if doc_count >= TARGET_DOC_COUNT_MIN else ' '}] Replace toy corpus with 20-50 real documents")
    print(f"[{'x' if question_count >= TARGET_QUESTION_COUNT else ' '}] Write {TARGET_QUESTION_COUNT} hand-written eval questions")
    print("[ ] Run baseline retrieval: python src/build_index.py")
    print("[ ] Run baseline evaluation: python src/evaluate_retrieval.py")
    print("[ ] Run baseline generation: python src/ask.py \"your question\"")
    print("[ ] Log failures in notes/failure_cases.md")
    print("[ ] Log small experiments in experiments/")
    print()

    if DATASET_STAGE == "toy":
        print("Next step")
        print("-" * 40)
        print("Add 20-50 documents to data/raw/ and hand-written eval questions.")
        print("See data/README.md for dataset options.")
    elif doc_count < TARGET_DOC_COUNT_MIN:
        print("Next step")
        print("-" * 40)
        print(f"Add {TARGET_DOC_COUNT_MIN - doc_count} more documents to reach the Part 21 minimum.")
    print("=" * 72)


if __name__ == "__main__":
    main()
