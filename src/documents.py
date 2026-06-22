from pathlib import Path

from pypdf import PdfReader

from config import RAW_DIR

SUPPORTED_SUFFIXES = {".txt", ".pdf"}


def is_document_file(path):
    if not path.is_file():
        return False
    if ":Zone.Identifier" in path.name:
        return False
    return path.suffix.lower() in SUPPORTED_SUFFIXES


def list_raw_documents():
    paths = [p for p in RAW_DIR.iterdir() if is_document_file(p)]
    return sorted(paths, key=lambda p: p.name.lower())


def load_txt(path):
    return path.read_text(encoding="utf-8")


def load_pdf(path):
    reader = PdfReader(path)
    pages = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())

    return "\n\n".join(pages)


def load_document(path):
    suffix = path.suffix.lower()

    if suffix == ".txt":
        text = load_txt(path)
    elif suffix == ".pdf":
        text = load_pdf(path)
    else:
        raise ValueError(f"Unsupported document type: {path.name}")

    return text.strip()


def load_raw_documents():
    documents = []
    skipped = []

    for path in list_raw_documents():
        try:
            text = load_document(path)
        except Exception as exc:
            skipped.append((path.name, str(exc)))
            continue

        if not text:
            skipped.append((path.name, "no extractable text"))
            continue

        documents.append({
            "source": path.name,
            "format": path.suffix.lower().lstrip("."),
            "text": text,
        })

    return documents, skipped
