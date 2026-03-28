#!/usr/bin/env python3
import json
from collections import defaultdict
from pathlib import Path


LABEL_MAP = {
    "Person": "person",
    "Location": "location",
    "Organization": "organization",
}


def convert_record(record: dict) -> dict:
    text = record["text"]
    entities = defaultdict(list)

    for span in record.get("spans", []):
        start = span["start"]
        end = span["end"]
        label = LABEL_MAP.get(span["label"], span["label"].lower())
        mention = text[start:end].strip()
        if mention and mention not in entities[label]:
            entities[label].append(mention)

    return {
        "input": text,
        "output": {
            "entities": dict(entities),
        },
    }


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / "dd_corpus_small_train.json"
    output_path = base_dir / "dd_corpus_small_train.jsonl"

    records = json.loads(input_path.read_text(encoding="utf-8"))
    converted = [convert_record(record) for record in records]

    with output_path.open("w", encoding="utf-8") as f:
        for record in converted:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Converted {len(converted)} records to {output_path}")


if __name__ == "__main__":
    main()
