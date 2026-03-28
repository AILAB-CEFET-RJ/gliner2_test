#!/usr/bin/env python3
import argparse
import json
import tempfile
from pathlib import Path

from gliner2 import GLiNER2
from gliner2.training.data import TrainingDataset
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig


LABEL_MAP = {
    "Person": "person",
    "Location": "location",
    "Organization": "organization",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a GLiNER2 model on the NERDD dataset with full fine-tuning."
    )
    parser.add_argument(
        "--data",
        default="nerdd/dd_corpus_small_train.jsonl",
        help="Path to the training dataset in GLiNER2 JSONL format.",
    )
    parser.add_argument(
        "--model",
        default="fastino/gliner2-base-v1",
        help="Base GLiNER2 model name or local path.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/nerdd_full",
        help="Directory for checkpoints and final full fine-tuned model.",
    )
    parser.add_argument(
        "--experiment-name",
        default="nerdd_full",
        help="Experiment name used in logs and saved config.",
    )
    parser.add_argument("--num-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--encoder-lr", type=float, default=5e-6)
    parser.add_argument("--task-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--keep-empty-examples",
        action="store_true",
        help="Keep examples with no annotated entities.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable fp16 mixed precision. Recommended when training on GPU.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce memory usage.",
    )
    return parser.parse_args()


def convert_record(record: dict) -> dict:
    if "input" in record and "output" in record:
        return record

    text = record["text"]
    entities: dict[str, list[str]] = {}

    for span in record.get("spans", []):
        start = span["start"]
        end = span["end"]
        label = LABEL_MAP.get(span["label"], span["label"].lower())
        mention = text[start:end].strip()
        if mention:
            entities.setdefault(label, [])
            if mention not in entities[label]:
                entities[label].append(mention)

    return {
        "input": text,
        "output": {"entities": entities},
    }


def load_dataset(data_path: Path, seed: int) -> tuple[TrainingDataset, Path | None]:
    temp_jsonl_path: Path | None = None
    load_path = data_path

    if data_path.suffix == ".json":
        records = json.loads(data_path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"Expected a JSON array in {data_path}")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", encoding="utf-8", delete=False
        ) as temp_file:
            for record in records:
                converted = convert_record(record)
                temp_file.write(json.dumps(converted, ensure_ascii=False) + "\n")
            temp_jsonl_path = Path(temp_file.name)

        load_path = temp_jsonl_path

    dataset = TrainingDataset.load(load_path, shuffle=True, seed=seed)
    return dataset, temp_jsonl_path


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    temp_jsonl_path: Path | None = None

    try:
        dataset, temp_jsonl_path = load_dataset(data_path, args.seed)

        if not args.keep_empty_examples:
            dataset = dataset.filter(lambda ex: bool(ex.entities))

        train_data, val_data, _ = dataset.split(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=1.0 - args.train_ratio - args.val_ratio,
            shuffle=True,
            seed=args.seed,
        )

        if len(train_data.examples) == 0:
            raise ValueError("Training split is empty.")
        if len(val_data.examples) == 0:
            raise ValueError("Validation split is empty.")

        model = GLiNER2.from_pretrained(args.model)

        config = TrainingConfig(
            output_dir=args.output_dir,
            experiment_name=args.experiment_name,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.grad_accum,
            encoder_lr=args.encoder_lr,
            task_lr=args.task_lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            logging_steps=args.logging_steps,
            eval_strategy="epoch",
            eval_steps=args.eval_steps,
            save_best=True,
            early_stopping=True,
            early_stopping_patience=2,
            validate_data=True,
            fp16=args.fp16,
            gradient_checkpointing=args.gradient_checkpointing,
            use_lora=False,
            seed=args.seed,
        )

        trainer = GLiNER2Trainer(model=model, config=config)
        results = trainer.train(train_data=train_data, eval_data=val_data)

        print("Training completed.")
        print(f"Train examples: {len(train_data.examples)}")
        print(f"Validation examples: {len(val_data.examples)}")
        print(f"Best metric: {results.get('best_metric')}")
        print(f"Output dir: {args.output_dir}")
    finally:
        if temp_jsonl_path and temp_jsonl_path.exists():
            temp_jsonl_path.unlink()


if __name__ == "__main__":
    main()
