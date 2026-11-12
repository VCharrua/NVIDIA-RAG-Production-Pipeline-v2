import argparse
import csv
from pathlib import Path

from .config import settings
from .indexing import build_index
from .pipeline import RAGPipeline


def run_submission(output: Path) -> None:
    pipeline = RAGPipeline(settings)
    output.parent.mkdir(parents=True, exist_ok=True)
    with settings.queries_path.open(newline="", encoding="utf-8") as source, output.open(
        "w", newline="", encoding="utf-8"
    ) as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(
            target,
            fieldnames=["query_id", "retrieved_chunk_ids", "generated_answer"],
        )
        writer.writeheader()
        for row in reader:
            result = pipeline.answer(row["query_text"])
            writer.writerow(
                {
                    "query_id": row["query_id"],
                    "retrieved_chunk_ids": "|".join(result["retrieved_chunk_ids"]),
                    "generated_answer": result["answer"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="NVIDIA NIM benchmark RAG pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("index", help="Build the FAISS index")
    run_parser = subparsers.add_parser("run", help="Generate benchmark submission.csv")
    run_parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()
    if args.command == "index":
        build_index(settings)
    else:
        run_submission(args.output)
