from dataclasses import dataclass
from typing import Any

from langchain_nvidia_ai_endpoints import ChatNVIDIA

from .config import Settings
from .indexing import load_index

SYSTEM_PROMPT = """You answer questions using only the supplied context.
If the context does not contain enough evidence, reply exactly:
Information not present in context.
Do not use outside knowledge. Preserve numerical units and caveats.
Cite supporting chunks inline as [chunk_XXXX].
"""


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    content: str
    score: float
    metadata: dict[str, Any]


class RAGPipeline:
    def __init__(self, config: Settings):
        self.config = config
        self.index = load_index(config)
        llm_kwargs: dict[str, Any] = {
            "model": config.nvidia_llm_model,
            "base_url": config.nvidia_llm_base_url,
            "temperature": 0.1,
            "max_completion_tokens": 1000,
            "top_p": 1.0,
        }
        if config.nvidia_api_key:
            llm_kwargs["nvidia_api_key"] = config.nvidia_api_key
        self.llm = ChatNVIDIA(**llm_kwargs)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        if not query.strip():
            raise ValueError("Query must not be empty.")
        pairs = self.index.similarity_search_with_relevance_scores(
            query, k=top_k or self.config.rag_top_k
        )
        return [
            RetrievedChunk(
                chunk_id=str(document.metadata["chunk_id"]),
                content=document.page_content,
                score=float(score),
                metadata=document.metadata,
            )
            for document, score in pairs
            if score >= self.config.rag_min_score
        ]

    def answer(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        chunks = self.retrieve(query, top_k)
        if not chunks:
            return {
                "answer": "Information not present in context.",
                "retrieved_chunk_ids": [],
                "chunks": [],
            }
        context = "\n\n".join(
            f"[{chunk.chunk_id}] {chunk.content}" for chunk in chunks
        )
        response = self.llm.invoke(
            [
                ("system", SYSTEM_PROMPT),
                ("human", f"Context:\n{context}\n\nQuestion: {query}"),
            ]
        )
        return {
            "answer": str(response.content),
            "retrieved_chunk_ids": [chunk.chunk_id for chunk in chunks],
            "chunks": [chunk.__dict__ for chunk in chunks],
        }
