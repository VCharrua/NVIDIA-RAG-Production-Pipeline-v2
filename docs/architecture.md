# NVIDIA NIM RAG pipeline

The project follows the NVIDIA LangChain endpoint pattern, but points both
model clients at local OpenAI-compatible NIM services:

1. `index` reads the benchmark's canonical `corpus_chunks.jsonl` passages.
   These passages are already the official retrieval units, so their
   `chunk_XXXX` IDs are preserved exactly for leaderboard scoring.
2. `NVIDIAEmbeddings` calls the local NeMo Retriever embedding NIM and stores
   vectors plus metadata in FAISS.
3. Retrieval uses similarity scores and a configurable minimum score. The
   prompt includes chunk IDs, document metadata, and the explicit refusal
   instruction recommended for hallucination control.
4. `ChatNVIDIA` calls the local LLM NIM. No commercial or hosted model API is
   required at runtime.
5. `run` processes every query and writes the required
   `query_id,retrieved_chunk_ids,generated_answer` CSV.

The Compose file starts separate LLM and embedding NIM containers and a small
FastAPI service. Set `NGC_API_KEY` only in the shell or an uncommitted `.env`;
never commit it.
