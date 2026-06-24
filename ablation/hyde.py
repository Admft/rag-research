"""HyDE query transformation for Ablation 7.

The live pipeline applies HyDE when query_transform == "hyde" via
src/retrievers.py transform_query(). This module provides the same
transform as a standalone function for documentation and reuse.
"""

from llm import call_ollama


def hyde_transform(query: str, llm_model: str) -> str:
    """
    Generate a hypothetical answer to the query using the LLM,
    then use that hypothetical answer as the retrieval query.
    The hypothetical answer looks more like document text than
    the question does, improving dense retrieval.
    """
    prompt = (
        "Write a short paragraph that would answer the following "
        "question. You may speculate — factual accuracy is not "
        "required. Just write what an answer might look like.\n\n"
        f"Question: {query}\n\nHypothetical answer:"
    )
    text, _ = call_ollama(prompt, model=llm_model)
    return text.strip() or query
