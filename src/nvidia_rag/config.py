from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for local NIMs and the benchmark corpus."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    rag_data_dir: Path = Path("kaggle/input/rag-data")
    rag_index_dir: Path = Path("artifacts/index")
    nvidia_llm_base_url: str = "http://localhost:8000/v1"
    nvidia_embedding_base_url: str = "http://localhost:8001/v1"
    nvidia_llm_model: str = "meta/llama-3.1-8b-instruct"
    nvidia_embedding_model: str = "nvidia/nv-embedqa-e5-v5"
    nvidia_api_key: str | None = None
    rag_top_k: int = 8
    rag_min_score: float = 0.15
    chunk_size: int = 600

    @property
    def chunks_path(self) -> Path:
        return self.rag_data_dir / "corpus" / "corpus_chunks.jsonl"

    @property
    def queries_path(self) -> Path:
        return self.rag_data_dir / "test_suite" / "test_queries.csv"


settings = Settings()
