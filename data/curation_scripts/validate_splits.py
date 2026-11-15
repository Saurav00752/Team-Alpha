import argparse
import hashlib
import json
import os
import sys
from typing import Dict, List, Set, Tuple
import yaml
from transformers import AutoTokenizer


REQUIRED_ROLES = {"system", "user", "assistant"}


def load_jsonl(file_path: str) -> List[Dict]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Split file missing: {file_path}")
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {file_path} at line {idx}: {e}")
    return rows


def validate_schema_and_hashes(rows: List[Dict], split_name: str) -> Tuple[Set[str], Set[str]]:
    """Verifies message structure and returns full-pair & user-prompt hashes for leakage checks."""
    pair_hashes = set()
    user_hashes = set()

    for idx, row in enumerate(rows, start=1):
        if "messages" not in row or not isinstance(row["messages"], list):
            raise ValueError(f"[{split_name}:{idx}] Missing 'messages' list.")

        roles_present = [m.get("role") for m in row["messages"]]
        if not REQUIRED_ROLES.issubset(set(roles_present)):
            raise ValueError(f"[{split_name}:{idx}] Missing required roles. Found: {roles_present}")

        if row["messages"][-1].get("role") != "assistant":
            raise ValueError(f"[{split_name}:{idx}] Final message must have role='assistant'.")

        for m in row["messages"]:
            content = m.get("content", "")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"[{split_name}:{idx}] Empty content in role '{m.get('role')}'.")

        user_text = next(m["content"].strip().lower() for m in row["messages"] if m["role"] == "user")
        asst_text = row["messages"][-1]["content"].strip().lower()

        u_hash = hashlib.sha256(user_text.encode("utf-8")).hexdigest()
        p_hash = hashlib.sha256(f"{user_text}|||{asst_text}".encode("utf-8")).hexdigest()
        user_hashes.add(u_hash)
        pair_hashes.add(p_hash)

    return pair_hashes, user_hashes


def check_token_lengths(rows: List[Dict], tokenizer, max_seq_length: int, split_name: str) -> None:
    """Computes tokenized length using the exact model chat template to detect truncation."""
    lengths = []
    over_limit = 0

    for row in rows:
        token_ids = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=True,
            add_generation_prompt=False,
        )
        # Handle nested list return if returned by specific tokenizer versions
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        seq_len = len(token_ids)
        lengths.append(seq_len)
        if seq_len > max_seq_length:
            over_limit += 1

    lengths.sort()
    p50 = lengths[len(lengths) // 2]
    p95 = lengths[int(len(lengths) * 0.95)]
    max_len = lengths[-1]
    pct_truncated = (over_limit / len(lengths)) * 100

    print(
        f"  -> {split_name:<12} | Count: {len(lengths):<5} | "
        f"Median: {p50:<4} tok | P95: {p95:<4} tok | Max: {max_len:<4} tok | "
        f"Exceeds {max_seq_length}: {over_limit} ({pct_truncated:.1f}%)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate JSONL schema, token lengths, and zero test leakage.")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Directory containing split JSONL files")
    parser.add_argument("--config", type=str, default="training/configs/config.yaml", help="Path to training config")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_id = cfg["model"]["base_model_id"]
    max_seq_length = cfg["model"]["max_seq_length"]

    splits = {
        "train": load_jsonl(os.path.join(args.data_dir, "train.jsonl")),
        "val": load_jsonl(os.path.join(args.data_dir, "val.jsonl")),
        "test": load_jsonl(os.path.join(args.data_dir, "test.jsonl")),
    }

    print("[STEP 1] Validating JSONL schema & computing hashes...")
    hashes = {}
    for name, rows in splits.items():
        hashes[name] = validate_schema_and_hashes(rows, name)

    print("[STEP 2] Auditing cross-split data leakage...")
    train_pairs, train_users = hashes["train"]
    val_pairs, val_users = hashes["val"]
    test_pairs, test_users = hashes["test"]

    test_train_pair_leak = len(test_pairs & train_pairs)
    test_val_pair_leak = len(test_pairs & val_pairs)
    test_train_prompt_leak = len(test_users & train_users)

    if test_train_pair_leak > 0 or test_val_pair_leak > 0:
        print(
            f"[FAIL] Exact pair leakage detected! "
            f"Test-Train overlap: {test_train_pair_leak}, Test-Val overlap: {test_val_pair_leak}"
        )
        sys.exit(1)

    if test_train_prompt_leak > 0:
        print(f"[WARN] {test_train_prompt_leak} identical user prompts appear in both train and test splits.")
    else:
        print("  -> PASS: 0% exact pair or prompt overlap between test.jsonl and train/val splits.")

    print(f"[STEP 3] Auditing token length distribution against '{model_id}' (max_seq_length={max_seq_length})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    for name, rows in splits.items():
        check_token_lengths(rows, tokenizer, max_seq_length, name)

    print("\n[SUCCESS] All dataset checks passed. Ready for QLoRA training.")


if __name__ == "__main__":
    main()