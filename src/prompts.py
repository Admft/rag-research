PROMPT_TEMPLATES = {
    "basic": """Answer the question using the provided context.

Context:
{context}

Question:
{query}

Answer:
""",
    "strict_context": """Answer the question using only the provided context.
If the context does not contain enough information, say: "I don't know based on the provided documents."

Context:
{context}

Question:
{query}

Answer:
""",
    "strict_context_with_citations": """Answer the question using only the provided context.
Cite the chunk IDs you used, like [chunk 0 from source.pdf].
If the answer is not supported by the context, say you do not know.

Context:
{context}

Question:
{query}

Answer:
""",
}


def format_context(retrieved):
    blocks = []
    for i, item in enumerate(retrieved, start=1):
        blocks.append(
            f"[chunk {item['chunk_index']} from {item['source']}]\n"
            f"{item['text']}"
        )
    return "\n\n".join(blocks)


def build_generation_prompt(query, retrieved, prompt_style):
    template = PROMPT_TEMPLATES[prompt_style]
    return template.format(
        context=format_context(retrieved),
        query=query,
    )
