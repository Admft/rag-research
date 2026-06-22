def chunk_text(text, chunk_size_words, overlap_words):
    words = text.split()
    chunks = []

    if overlap_words >= chunk_size_words:
        raise ValueError("overlap must be smaller than chunk size")

    start = 0
    while start < len(words):
        end = start + chunk_size_words
        chunk_words = words[start:end]
        chunk_text_value = " ".join(chunk_words)

        if chunk_text_value.strip():
            chunks.append(chunk_text_value)

        start += chunk_size_words - overlap_words

    return chunks
