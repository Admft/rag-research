import re

PROMPT_TEMPLATES = {
    "basic": """Answer the question using the provided documents.

<documents>
{context}
</documents>

Question:
{query}

Answer:
""",
    "strict_context": """Answer the question using only the provided documents.
If the documents do not contain enough information, say: "The provided context does not contain the answer."

Citation rules:
- Cite every factual claim with [Doc X] where X is the Document ID.
- Use one document per bracket. Do not combine citations.

<documents>
{context}
</documents>

Question:
{query}

Answer:
""",
    "strict_context_with_citations": """You are an expert analytical assistant. Your task is to answer the user's question accurately, strictly using ONLY the provided documents.

<documents>
{context}
</documents>

Follow these strict instructions:
1. You must first use a <scratchpad> block to extract relevant facts from the documents and explicitly map them to their corresponding [Doc X] IDs.
2. After your scratchpad, provide your final response inside an <answer> block.
3. Every factual claim in your <answer> MUST be immediately followed by its exact source citation using the format [Doc X].
4. Do not combine citations. If a sentence uses facts from Document 1 and Document 3, write "[Doc 1] [Doc 3]", NOT "[Doc 1, 3]".
5. Do not include introductory filler in your final answer.
6. If the provided documents do not contain sufficient information to answer the question, state exactly in <answer>: "The provided context does not contain the answer."

Begin.

Question:
{query}
""",
}


def format_context(retrieved):
    blocks = []
    for i, item in enumerate(retrieved, start=1):
        blocks.append(
            f'<Document ID="{i}" source="{item["source"]}" chunk="{item["chunk_index"]}">\n'
            f"{item['text']}\n"
            f"</Document>"
        )
    return "\n\n".join(blocks)


def extract_final_answer(text):
    if not text:
        return ""

    closed = re.search(r"<answer>\s*(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if closed:
        return closed.group(1).strip()

    unclosed = re.search(r"<answer>\s*(.*)$", text, re.DOTALL | re.IGNORECASE)
    if unclosed:
        return unclosed.group(1).strip()

    if re.search(r"<scratchpad>", text, re.IGNORECASE):
        parts = re.split(r"</scratchpad>", text, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) > 1:
            remainder = re.sub(r"^<answer>\s*", "", parts[1].strip(), flags=re.IGNORECASE)
            remainder = re.sub(r"</answer>\s*$", "", remainder, flags=re.IGNORECASE)
            if remainder:
                return remainder.strip()

    legacy = re.search(r"<Answer>\s*(.*?)(?:</Answer>|$)", text, re.DOTALL | re.IGNORECASE)
    if legacy:
        return legacy.group(1).strip()

    if re.search(r"<scratchpad>|here are the facts|planning", text, re.IGNORECASE):
        return ""

    return text.strip()


def has_answer_block(text):
    return bool(re.search(r"<answer>", text, re.IGNORECASE))


def build_generation_prompt(query, retrieved, prompt_style):
    template = PROMPT_TEMPLATES[prompt_style]
    return template.format(
        context=format_context(retrieved),
        query=query,
    )
