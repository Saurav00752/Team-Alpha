import argparse
import csv
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_SYSTEM_PROMPT = (
    "You are a specialized domain expert. Analyze the provided input accurately "
    "and produce a structured, technically valid response following domain standards."
)


def compute_record_hash(user_text: str, assistant_text: str) -> str:
    """Computes a normalized SHA-256 hash to eliminate exact/whitespace duplicates."""
    normalized = f"{user_text.strip().lower()}|||{assistant_text.strip().lower()}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_raw_records(input_path: str) -> List[Dict[str, str]]:
    """Loads raw records from .jsonl, .json, or .csv files."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw input file not found: {input_path}")

    records = []
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[WARN] Skipping malformed JSON at line {line_num}: {e}")

    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON root must be a list of objects.")
            records.extend(data)

    elif suffix == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            records.extend(list(reader))

    else:
        raise ValueError(f"Unsupported file format '{suffix}'. Use .jsonl, .json, or .csv.")

    return records


def normalize_to_messages(
    raw_records: List[Dict[str, str]],
    input_col: str,
    output_col: str,
    system_prompt: str,
) -> Tuple[List[Dict], int, int]:
    """Converts raw rows into TRL SFTTrainer 'messages' schema and deduplicates."""
    seen_hashes = set()
    formatted = []
    skipped_empty = 0
    skipped_duplicates = 0

    for row in raw_records:
        # Support pre-formatted messages if present, otherwise map input/output columns
        if "messages" in row and isinstance(row["messages"], list):
            msgs = row["messages"]
            user_msgs = [m["content"] for m in msgs if m.get("role") == "user"]
            asst_msgs = [m["content"] for m in msgs if m.get("role") == "assistant"]
            if not user_msgs or not asst_msgs:
                skipped_empty += 1
                continue
            user_text = user_msgs[-1].strip()
            assistant_text = asst_msgs[-1].strip()
            sys_text = next(
                (m["content"].strip() for m in msgs if m.get("role") == "system"),
                system_prompt,
            )
        else:
            user_text = str(row.get(input_col, "")).strip()
            assistant_text = str(row.get(output_col, "")).strip()
            sys_text = str(row.get("system", system_prompt)).strip() or system_prompt

        if not user_text or not assistant_text:
            skipped_empty += 1
            continue

        rec_hash = compute_record_hash(user_text, assistant_text)
        if rec_hash in seen_hashes:
            skipped_duplicates += 1
            continue
        seen_hashes.add(rec_hash)

        formatted.append(
            {
                "messages": [
                    {"role": "system", "content": sys_text},
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ]
            }
        )

    return formatted, skipped_empty, skipped_duplicates


def split_and_save(
    records: List[Dict],
    output_dir: str,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> None:
    """Shuffles deterministically and writes train.jsonl, val.jsonl, and test.jsonl."""
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)
    rng.shuffle(records)

    total = len(records)
    if total < 10:
        raise ValueError(f"Dataset too small ({total} valid records). Need at least 10 rows.")

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train.jsonl": records[:train_end],
        "val.jsonl": records[train_end:val_end],
        "test.jsonl": records[val_end:],
    }

    for filename, subset in splits.items():
        out_path = os.path.join(output_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            for item in subset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[SAVED] {out_path} -> {len(subset)} records")


def main() -> None:
    parser = argparse.ArgumentParser(description="Format raw domain data into QLoRA JSONL splits.")
    parser.add_argument("--input", type=str, required=True, help="Path to raw .jsonl, .json, or .csv file")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output folder for splits")
    parser.add_argument("--input-col", type=str, default="input", help="Column/key name for user prompt")
    parser.add_argument("--output-col", type=str, default="output", help="Column/key name for expert target")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT, help="Domain system prompt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splitting")
    args = parser.parse_args()

    raw_records = load_raw_records(args.input)
    print(f"[INFO] Loaded {len(raw_records)} raw records from {args.input}")

    cleaned, skipped_empty, skipped_dups = normalize_to_messages(
        raw_records=raw_records,
        input_col=args.input_col,
        output_col=args.output_col,
        system_prompt=args.system_prompt,
    )
    print(
        f"[INFO] Cleaned: {len(cleaned)} valid | "
        f"{skipped_empty} empty/invalid dropped | {skipped_dups} duplicates removed"
    )

    split_and_save(cleaned, output_dir=args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()