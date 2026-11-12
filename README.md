# NVIDIA RAG Test Pipeline

An NVIDIA NIM-powered Retrieval-Augmented Generation (RAG) system for the
**Advanced Local RAG Benchmark**. The project adapts the benchmark objective
to LangChain's NVIDIA integrations, FAISS, NVIDIA NeMo Retriever embeddings,
and an OpenAI-compatible NVIDIA NIM chat service.

The pipeline is designed for technical enterprise information retrieval across
naval engineering specifications, contracts, regulatory documents, maintenance
manuals, fleet ledgers, and incident records. It can:

- retrieve authoritative passages from the supplied enterprise corpus;
- preserve the benchmark's canonical `chunk_XXXX` identifiers;
- retain document, section, clause, table, and cross-reference metadata;
- answer questions using only retrieved evidence;
- refuse unsupported questions instead of inventing facts;
- generate the required `submission.csv` for all 200 benchmark queries;
- run behind a local FastAPI service using NVIDIA GPU containers.

The full benchmark requirements are documented in
[`docs/project-specifications.md`](docs/project-specifications.md).

## Repository contents

```text
.
├── docs/
│   ├── architecture.md
│   └── project-specifications.md
├── kaggle/
│   └── input/rag-data/
├── scripts/
│   └── validate_submission.py
├── src/nvidia_rag/
│   ├── api.py
│   ├── cli.py
│   ├── config.py
│   ├── indexing.py
│   └── pipeline.py
├── .env.example
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## NVIDIA pipeline

### Canonical corpus indexing

The indexer reads `corpus_chunks.jsonl` directly instead of generating new
IDs. This is intentional: the supplied passages are the official retrieval
units used by the benchmark. Each passage is stored in FAISS with its
`chunk_id`, document identity, section, parent clause, page, table flag, and
structured metadata.

### NVIDIA embeddings

`NVIDIAEmbeddings` calls the NVIDIA NeMo Retriever embedding NIM. The resulting
vectors are persisted locally under `artifacts/index`, so indexing is
performed once and reused for subsequent queries.

### NVIDIA generation

`ChatNVIDIA` calls a local NVIDIA NIM through its OpenAI-compatible `/v1`
endpoint. The prompt requires evidence-only answers, preserves numerical units,
and requires inline citations such as `[chunk_0042]`.

The same LangChain clients can target NVIDIA AI Endpoints by setting
`NVIDIA_API_KEY` and changing the base URLs. Local NIM mode is the default
deployment path.

## Quick start

Create the Python environment and configure the endpoints:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
Copy-Item .env.example .env
```

Start the NVIDIA NIM services with an NGC credential:

```powershell
$env:NGC_API_KEY = "..."
docker compose up -d llm embedding
```

Build the FAISS index and generate the benchmark submission:

```powershell
nvidia-rag index
nvidia-rag run --output submission.csv
python scripts\validate_submission.py submission.csv
```

The generated file contains exactly:

```text
query_id,retrieved_chunk_ids,generated_answer
```

## Docker deployment

The [Dockerfile](Dockerfile) uses
`nvidia/cuda:12.4.1-runtime-ubuntu22.04`. The Compose deployment reserves an
NVIDIA GPU for the LLM NIM, embedding NIM, and API services:

```powershell
$env:NGC_API_KEY = "..."
docker compose up --build
```

The default NIM images are configurable:

```powershell
$env:NIM_LLM_IMAGE = "nvcr.io/nvidia/llama-3.1-8b-instruct:1.3.3"
$env:NIM_EMBEDDING_IMAGE = "nvcr.io/nvidia/nv-embedqa-e5-v5:1.5.0"
docker compose up -d llm embedding
```

Use the tags available in the target NGC registry and compatible with the
installed NVIDIA driver and CUDA runtime. The NGC credential should remain in
the shell or an uncommitted `.env` file.

The API is exposed at port `8080`:

```powershell
Invoke-RestMethod http://localhost:8080/health
Invoke-RestMethod -Method Post http://localhost:8080/query `
  -ContentType "application/json" `
  -Body '{"query":"What is the continuous maximum power rating of the GTX-9000?"}'
```

The response includes the grounded answer, retrieved canonical chunk IDs, and
the supporting passages. The service uses one application process so local
model state is not unnecessarily duplicated across GPU workers.

## Dataset

The supplied corpus is mounted at:

```text
/kaggle/input/rag-data/
├── corpus/
│   ├── corpus_chunks.jsonl
│   ├── corpus_manifest.json
│   └── raw_documents/
└── test_suite/
    └── test_queries.csv
```

The repository includes the same layout under
[`kaggle/input/rag-data`](kaggle/input/rag-data). The canonical chunks file
contains 2,584 passages and the test suite contains 200 query scenarios.

## Evaluation and grounding controls

The implementation targets the benchmark's retrieval and generation
requirements with:

- FAISS dense retrieval using NVIDIA NeMo Retriever embeddings;
- configurable top-k retrieval and minimum relevance score;
- canonical chunk ID preservation for MAP@K scoring;
- metadata attached to every retrieved passage;
- explicit context-only generation instructions;
- numerical-unit preservation;
- deterministic refusal when no passage clears the relevance threshold;
- CSV generation and validation for all benchmark queries.

The project intentionally does not claim hybrid lexical retrieval,
cross-encoder reranking, or local model quantization in this implementation.
Those are possible future extensions, but the current runtime is centered on
the NVIDIA LangChain and NIM workflow described in the supplied NVIDIA guide.

## Air-gapped operating model

Runtime inference is local when using the NIM Compose deployment; it does not
call OpenAI, Anthropic, Google Gemini, Cohere, or another commercial inference
API. Internet access is needed only to provision Python packages and pull
approved NIM images. For an air-gapped deployment:

1. pull and verify the NIM images before entering the enclave;
2. pre-populate the NIM model cache;
3. transfer the image and corpus into the target environment;
4. expose only the internal API port;
5. run the LLM and embedding services with the approved NVIDIA runtime.

## Development checks

Static checks that do not require NIM services or GPU inference:

```powershell
python -m compileall -q src scripts
git --no-pager diff --check
docker compose config --quiet
```

Full embedding, retrieval, and generation validation requires NVIDIA GPU
hardware, compatible drivers, NIM images, and an available model service.
