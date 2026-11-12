from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import settings
from .pipeline import RAGPipeline

app = FastAPI(title="NVIDIA NIM RAG Pipeline", version="0.1.0")
pipeline: RAGPipeline | None = None


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query")
def query(request: QueryRequest) -> dict:
    global pipeline
    try:
        if pipeline is None:
            pipeline = RAGPipeline(settings)
        return pipeline.answer(request.query, request.top_k)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
