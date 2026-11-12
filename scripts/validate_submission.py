import argparse
import csv
from pathlib import Path


EXPECTED_COLUMNS = ["query_id", "retrieved_chunk_ids", "generated_answer"]


def validate(path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(
                f"Columns mismatch: expected {EXPECTED_COLUMNS}, got {reader.fieldnames}"
            )
        rows = list(reader)

    if len(rows) != 200:
        raise ValueError(f"Expected exactly 200 query rows, found {len(rows)}")
    query_ids = [row["query_id"] for row in rows]
    if any(not query_id for query_id in query_ids):
        raise ValueError("Found an empty query_id")
    if len(set(query_ids)) != 200:
        raise ValueError("Duplicate query_id entries detected")
    if not any(row["generated_answer"].strip() for row in rows):
        raise ValueError("Generated answers are completely empty")
    print(f"[SUCCESS] {path} is compliant with the benchmark CSV contract")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    validate(parser.parse_args().submission)
