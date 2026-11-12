import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from .config import Settings


def load_canonical_documents(path: Path) -> list[Document]:
    """Load official benchmark passages without changing their scoring IDs."""
    if not path.is_file():
        raise FileNotFoundError(f"Corpus file does not exist: {path}")

    documents: list[Document] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            chunk_id = record.get("chunk_id")
            content = record.get("content")
            if not isinstance(chunk_id, str) or not isinstance(content, str):
                raise ValueError(f"Missing chunk_id/content at {path}:{line_number}")
            metadata = {key: value for key, value in record.items() if key not in {"content"}}
            documents.append(Document(page_content=content, metadata=metadata))
    if not documents:
        raise ValueError(f"No canonical chunks found in {path}")
    return documents


def embedding_model(config: Settings) -> NVIDIAEmbeddings:
    kwargs: dict[str, str] = {
        "model": config.nvidia_embedding_model,
        "base_url": config.nvidia_embedding_base_url,
    }
    if config.nvidia_api_key:
        kwargs["nvidia_api_key"] = config.nvidia_api_key
    return NVIDIAEmbeddings(**kwargs)


def build_index(config: Settings) -> FAISS:
    """Create and persist a FAISS index using the NVIDIA NeMo Retriever NIM."""
    documents = load_canonical_documents(config.chunks_path)
    index = FAISS.from_documents(documents, embedding_model(config))
    config.rag_index_dir.mkdir(parents=True, exist_ok=True)
    index.save_local(str(config.rag_index_dir))
    return index


def load_index(config: Settings) -> FAISS:
    if not (config.rag_index_dir / "index.faiss").is_file():
        raise FileNotFoundError(
            f"FAISS index not found at {config.rag_index_dir}; run `nvidia-rag index` first."
        )
    return FAISS.load_local(
        str(config.rag_index_dir),
        embedding_model(config),
        # FAISS persists LangChain's local document metadata with pickle.
        # The path is controlled by this application and must not be user input.
        allow_dangerous_deserialization=True,
    )
