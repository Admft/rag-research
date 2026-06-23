import json
import re

from llm import parse_json_response

GENERATION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "scratchpad": {
            "type": "string",
            "description": "Brief planning notes with [Doc X] mappings, max 400 characters.",
        },
        "answer": {
            "type": "string",
            "description": "Final grounded answer. Cite inline using exact format [Doc 1], [Doc 2], etc.",
        },
        "citations": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Document numbers referenced in the answer, e.g. [1, 2].",
        },
    },
    "required": ["scratchpad", "answer", "citations"],
}

JSON_PROMPT_STYLES = {"strict_context_json"}

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
    "og_strict": """Answer the question using ONLY the provided context below.
If the context does not contain enough information to answer, say exactly: "I do not know."

Context:
{context}

Question:
{query}

Answer:
""",
    "strict_context_with_citations": """You are an expert analytical assistant. Your task is to answer the user's question accurately, strictly using ONLY the provided documents.

<documents>
{context}
</documents>

Follow these strict instructions:
1. You must first use a <scratchpad> block to extract relevant facts from the documents and explicitly map them to their corresponding [Doc X] IDs. Keep scratchpad notes brief: 3-4 short bullet points maximum.
2. After your scratchpad, provide your final response inside an <answer> block.
3. Every factual claim in your <answer> MUST be immediately followed by its exact source citation using the format [Doc X].
4. Do not combine citations. If a sentence uses facts from Document 1 and Document 3, write "[Doc 1] [Doc 3]", NOT "[Doc 1, 3]".
5. Do not include introductory filler in your final answer.
6. If the provided documents do not contain sufficient information to answer the question, state exactly in <answer>: "The provided context does not contain the answer."

Begin.

Question:
{query}
""",
    "strict_context_json": """You are an expert analytical assistant. Answer using ONLY the provided documents.

<documents>
{context}
</documents>

Return a single JSON object with exactly three keys:
- "scratchpad": 2-3 brief bullet points mapping facts to [Doc X] IDs. Hard limit: 400 characters total.
- "answer": final answer only. Every factual claim MUST be followed by [Doc X].
- "citations": array of document numbers cited in the answer (e.g. [1, 2]).

CRITICAL: Every citation in the answer field MUST use the exact format [Doc N] where N is the
document number. Do not write [Document N], [Document ID="N"], [Doc. N], or [N] alone.

If documents lack sufficient information, set answer to exactly:
"The provided context does not contain the answer." and citations to [].

Question:
{query}
""",
}


def format_context(retrieved, style="xml"):
    blocks = []
    for i, item in enumerate(retrieved, start=1):
        if style == "simple":
            blocks.append(
                f"[{i}] ({item['source']}, chunk {item['chunk_index']})\n"
                f"{item['text']}"
            )
        else:
            blocks.append(
                f'<Document ID="{i}" source="{item["source"]}" chunk="{item["chunk_index"]}">\n'
                f"{item['text']}\n"
                f"</Document>"
            )
    return "\n\n".join(blocks)


def normalize_citations(text):
    if not text:
        return text
    text = re.sub(r'\[Document ID=["\']?(\d+)["\']?\]', r"[Doc \1]", text, flags=re.IGNORECASE)
    text = re.sub(r"\[Document (\d+)\]", r"[Doc \1]", text, flags=re.IGNORECASE)
    text = re.sub(r"\[Doc\.\s*(\d+)\]", r"[Doc \1]", text, flags=re.IGNORECASE)
    text = re.sub(r"\[(\d+)\]", r"[Doc \1]", text)
    return text


def uses_json_output(prompt_style):
    return prompt_style in JSON_PROMPT_STYLES


def parse_generation_response(raw, prompt_style):
    if uses_json_output(prompt_style):
        try:
            data = parse_json_response(raw)
            answer = normalize_citations(str(data.get("answer", "")).strip())
            scratchpad = str(data.get("scratchpad", "")).strip()
            return answer, bool(answer), scratchpad
        except (ValueError, json.JSONDecodeError, TypeError):
            return "", False, ""

    answer = normalize_citations(extract_final_answer(raw))
    return answer, bool(answer), ""


def extract_final_answer(text):
    if not text:
        return ""

    if uses_json_output_from_text(text):
        try:
            data = parse_json_response(text)
            return normalize_citations(str(data.get("answer", "")).strip())
        except (ValueError, json.JSONDecodeError, TypeError):
            pass

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

    return normalize_citations(text.strip())


def uses_json_output_from_text(text):
    stripped = text.strip()
    return stripped.startswith("{") and '"answer"' in stripped


def has_answer_block(text, prompt_style=None):
    if prompt_style and uses_json_output(prompt_style):
        answer, parsed, _ = parse_generation_response(text, prompt_style)
        return parsed
    return bool(re.search(r"<answer>", text, re.IGNORECASE))


def build_generation_prompt(query, retrieved, prompt_style):
    template = PROMPT_TEMPLATES[prompt_style]
    context_style = "simple" if prompt_style == "og_strict" else "xml"
    return template.format(
        context=format_context(retrieved, style=context_style),
        query=query,
    )
