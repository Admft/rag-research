from config import COLLECTION_NAME
from llm import call_ollama


def reciprocal_rank_fusion(result_lists, top_k, k=60):
    scores = {}

    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            key = item["id"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    merged = {}
    for results in result_lists:
        for item in results:
            merged[item["id"]] = item

    ranked_ids = sorted(scores, key=lambda item_id: scores[item_id], reverse=True)
    return [
        {**merged[item_id], "score": scores[item_id]}
        for item_id in ranked_ids[:top_k]
    ]


def normalize_scores(results):
    if not results:
        return results

    scores = [item["score"] for item in results]
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        for item in results:
            item["score"] = 1.0
        return results

    for item in results:
        item["score"] = (item["score"] - min_score) / (max_score - min_score)

    return results


def merge_weighted(dense_results, bm25_results, top_k, alpha=0.5):
    dense = normalize_scores([dict(item) for item in dense_results])
    bm25 = normalize_scores([dict(item) for item in bm25_results])

    combined = {}
    for item in dense:
        combined[item["id"]] = {**item, "score": alpha * item["score"]}
    for item in bm25:
        if item["id"] in combined:
            combined[item["id"]]["score"] += (1 - alpha) * item["score"]
        else:
            combined[item["id"]] = {**item, "score": (1 - alpha) * item["score"]}

    ranked = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]


class Retriever:
    def __init__(self, index, config):
        self.index = index
        self.config = config
        self._reranker = None

    def _get_reranker(self):
        if self._reranker is not None:
            return self._reranker

        if self.config.reranker == "none":
            return None

        if self.config.reranker in {"cross_encoder", "bge"}:
            from sentence_transformers import CrossEncoder

            model_name = (
                "BAAI/bge-reranker-base"
                if self.config.reranker == "bge"
                else "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
            self._reranker = CrossEncoder(model_name)
            return self._reranker

        raise ValueError(f"Unsupported reranker: {self.config.reranker}")

    def transform_query(self, query):
        transform = self.config.query_transform

        if transform == "none":
            return [query]

        if transform == "query_rewrite":
            rewrite, _ = call_ollama(
                f"Rewrite this search query for document retrieval. "
                f"Return only the rewritten query.\n\nQuery: {query}"
            )
            return [rewrite.strip() or query]

        if transform == "hyde":
            hypo, _ = call_ollama(
                f"Write a short hypothetical passage that would answer this question. "
                f"Return only the passage.\n\nQuestion: {query}"
            )
            return [hypo.strip() or query]

        if transform == "multi_query":
            text, _ = call_ollama(
                "Generate 3 alternate search queries for the question below. "
                "Return one query per line, no numbering.\n\n"
                f"Question: {query}"
            )
            queries = [line.strip() for line in text.splitlines() if line.strip()]
            return queries[:3] or [query]

        if transform == "query_decomposition":
            text, _ = call_ollama(
                "Break this question into 2-3 simpler sub-questions for retrieval. "
                "Return one sub-question per line.\n\n"
                f"Question: {query}"
            )
            queries = [line.strip() for line in text.splitlines() if line.strip()]
            return queries[:3] or [query]

        raise ValueError(f"Unsupported query transform: {transform}")

    def _hyde_query(self, query):
        hypo, _ = call_ollama(
            f"Write a short hypothetical passage that would answer this question. "
            f"Return only the passage.\n\nQuestion: {query}"
        )
        return hypo.strip() or query

    def dense_search(self, query, top_k):
        vector = self.index.embed_model.encode(query, normalize_embeddings=True)
        results = self.index.qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector.tolist(),
            limit=top_k,
        )
        hits = []
        for point in results.points:
            hits.append({
                "id": str(point.id),
                "source": point.payload["source"],
                "chunk_index": point.payload["chunk_index"],
                "text": point.payload["text"],
                "score": float(point.score),
            })
        return hits

    def retrieve_for_query(self, query, top_k):
        retriever = self.config.retriever

        if retriever == "dense":
            return self.dense_search(query, top_k)

        if retriever == "bm25":
            return self.index.bm25_search(query, top_k)

        if retriever == "hybrid":
            dense = self.dense_search(query, top_k * 2)
            bm25 = self.index.bm25_search(query, top_k * 2)
            return merge_weighted(dense, bm25, top_k)

        if retriever == "hyde":
            hyde_query = self._hyde_query(query)
            return self.dense_search(hyde_query, top_k)

        if retriever == "hyde_hybrid":
            hyde_query = self._hyde_query(query)
            dense = self.dense_search(hyde_query, top_k * 2)
            bm25 = self.index.bm25_search(hyde_query, top_k * 2)
            return merge_weighted(dense, bm25, top_k)

        raise ValueError(f"Unsupported retriever: {retriever}")

    def rerank(self, query, results):
        reranker = self._get_reranker()
        if reranker is None or not results:
            return results

        pairs = [(query, item["text"]) for item in results]
        scores = reranker.predict(pairs)
        reranked = []
        for item, score in zip(results, scores):
            reranked.append({**item, "score": float(score)})
        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked

    def apply_context_filter(self, results):
        filt = self.config.context_filter
        if filt == "none":
            return results

        if filt.startswith("top_sentences_"):
            sentence_count = int(filt.split("_")[-1])
            filtered = []
            for item in results:
                sentences = [s.strip() for s in item["text"].split(".") if s.strip()]
                filtered.append({
                    **item,
                    "text": ". ".join(sentences[:sentence_count]) + (
                        "." if sentences[:sentence_count] else ""
                    ),
                })
            return filtered

        raise ValueError(f"Unsupported context filter: {filt}")

    def retrieve(self, query):
        fetch_k = self.config.top_k
        if self.config.reranker != "none":
            fetch_k = max(fetch_k * 2, 10)

        queries = self.transform_query(query)
        result_lists = [self.retrieve_for_query(q, fetch_k) for q in queries]

        if len(result_lists) == 1:
            results = result_lists[0]
        else:
            results = reciprocal_rank_fusion(result_lists, fetch_k)

        results = self.rerank(query, results)
        results = results[: self.config.top_k]
        return self.apply_context_filter(results)
