import argparse
import json
import os
import time
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from eval_baseline import compute_sample_metrics, load_jsonl


def run_finetuned_evaluation(
    model_id: str,
    adapter_dir: str,
    test_file: str,
    output_file: str,
    max_new_tokens: int = 256,
    limit: int = None,
) -> None:
    if not os.path.exists(adapter_dir):
        raise FileNotFoundError(
            f"Trained LoRA adapter not found at '{adapter_dir}'. "
            "Pass --adapter-dir pointing to your Google Drive or local adapter folder."
        )

    print(f"[INFO] Loading 4-bit Base Model '{model_id}' + QLoRA Adapter '{adapter_dir}'")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    test_records = load_jsonl(test_file)
    if limit:
        test_records = test_records[:limit]

    predictions = []
    total_em, total_f1, total_syntax, total_latency = 0.0, 0.0, 0.0, 0.0

    print(f"[INFO] Running QLoRA expert inference on {len(test_records)} test examples...")
    for idx, row in enumerate(test_records, start=1):
        messages = row["messages"]
        prompt_messages = [m for m in messages if m["role"] in ("system", "user")]
        reference = next(m["content"] for m in messages if m["role"] == "assistant")
        user_input = next(m["content"] for m in messages if m["role"] == "user")

        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        start_t = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        latency = time.perf_counter() - start_t

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        pred_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        scores = compute_sample_metrics(pred_text, reference)
        total_em += scores["exact_match"]
        total_f1 += scores["token_f1"]
        total_syntax += scores["syntax_valid"]
        total_latency += latency

        predictions.append(
            {
                "id": idx,
                "user_input": user_input,
                "reference": reference,
                "prediction": pred_text,
                "latency_sec": round(latency, 3),
                "metrics": scores,
            }
        )

        if idx % 10 == 0 or idx == len(test_records):
            print(f"  -> Processed {idx}/{len(test_records)} samples")

    n = len(test_records)
    summary = {
        "model_stage": "Fine-Tuned (QLoRA Domain Expert)",
        "base_model_id": model_id,
        "adapter_dir": adapter_dir,
        "num_samples": n,
        "aggregate_metrics": {
            "exact_match_pct": round((total_em / n) * 100, 2),
            "token_f1_pct": round((total_f1 / n) * 100, 2),
            "syntax_validity_pct": round((total_syntax / n) * 100, 2),
            "avg_latency_sec": round(total_latency / n, 3),
        },
        "samples": predictions,
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Fine-tuned metrics saved to {output_file}")
    print(json.dumps(summary["aggregate_metrics"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate QLoRA fine-tuned model on test.jsonl")
    parser.add_argument("--config", type=str, default="training/configs/config.yaml")
    parser.add_argument("--adapter-dir", type=str, default=None, help="Override final_adapter_dir from config.yaml")
    parser.add_argument("--test-file", type=str, default="data/processed/test.jsonl")
    parser.add_argument("--output", type=str, default="evaluation/results/finetuned_metrics.json")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    adapter_path = args.adapter_dir or cfg["training"]["final_adapter_dir"]

    run_finetuned_evaluation(
        model_id=cfg["model"]["base_model_id"],
        adapter_dir=adapter_path,
        test_file=args.test_file,
        output_file=args.output,
        max_new_tokens=args.max_new_tokens,
        li
      mit=args.limit,
      )
