import json
import os
import argparse
import yaml
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def plot_loss_curves(log_history: list, plot_path: str, json_path: str) -> None:
    """Extracts train & eval loss from Trainer logs and saves high-res report charts."""
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []

    for entry in log_history:
        if "loss" in entry and "step" in entry:
            train_steps.append(entry["step"])
            train_losses.append(entry["loss"])
        if "eval_loss" in entry and "step" in entry:
            eval_steps.append(entry["step"])
            eval_losses.append(entry["eval_loss"])

    # Save raw metrics JSON for reproducibility
    metrics_payload = {
        "train": [{"step": s, "loss": l} for s, l in zip(train_steps, train_losses)],
        "eval": [{"step": s, "eval_loss": l} for s, l in zip(eval_steps, eval_losses)],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    # Plot Training vs. Validation Loss Curve
    plt.figure(figsize=(10, 6), dpi=300)
    if train_steps:
        plt.plot(train_steps, train_losses, label="Training Loss", color="#2563eb", linewidth=2)
    if eval_steps:
        plt.plot(eval_steps, eval_losses, label="Validation Loss", color="#dc2626", linewidth=2, marker="o")

    plt.title("QLoRA Fine-Tuning: Training vs. Validation Loss", fontsize=14, fontweight="bold")
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Cross-Entropy Loss", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"[INFO] Loss curve saved to {plot_path}")


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    model_id = cfg["model"]["base_model_id"]

    print(f"[INFO] Loading tokenizer and 4-bit quantized base model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 1. Configure 4-bit NormalFloat (NF4) Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # 2. Configure LoRA Adapters
    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type="CAUSAL_LM",
        target_modules=lora_cfg["target_modules"],
    )

    # 3. Load JSONL Datasets (expects {"messages": [{"role": ..., "content": ...}]})
    dataset = load_dataset(
        "json",
        data_files={
            "train": cfg["data"]["train_file"],
            "validation": cfg["data"]["val_file"],
        },
    )

    # 4. Configure SFTTrainer
    t_cfg = cfg["training"]
    os.makedirs(t_cfg["output_dir"], exist_ok=True)
    os.makedirs(t_cfg["final_adapter_dir"], exist_ok=True)

    sft_config = SFTConfig(
        output_dir=t_cfg["output_dir"],
        num_train_epochs=t_cfg["num_train_epochs"],
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=t_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        learning_rate=t_cfg["learning_rate"],
        lr_scheduler_type=t_cfg["lr_scheduler_type"],
        warmup_ratio=t_cfg["warmup_ratio"],
        weight_decay=t_cfg["weight_decay"],
        optim=t_cfg["optim"],
        fp16=t_cfg["fp16"],
        bf16=t_cfg["bf16"],
        gradient_checkpointing=t_cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=t_cfg["logging_steps"],
        eval_strategy=t_cfg["eval_strategy"],
        eval_steps=t_cfg["eval_steps"],
        save_strategy=t_cfg["save_strategy"],
        save_steps=t_cfg["save_steps"],
        save_total_limit=t_cfg["save_total_limit"],
        max_seq_length=cfg["model"]["max_seq_length"],
        seed=t_cfg["seed"],
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print("[INFO] Starting QLoRA fine-tuning...")
    trainer.train()

    # 5. Save Final Trained LoRA Adapter (~50-100 MB) & Tokenizer
    print(f"[INFO] Saving final LoRA adapter to {t_cfg['final_adapter_dir']}")
    trainer.model.save_pretrained(t_cfg["final_adapter_dir"])
    tokenizer.save_pretrained(t_cfg["final_adapter_dir"])

    # 6. Export Loss Curves for GitHub & Final Report
    plot_loss_curves(
        log_history=trainer.state.log_history,
        plot_path=cfg["artifacts"]["loss_plot_path"],
        json_path=cfg["artifacts"]["metrics_json_path"],
    )
    print("[SUCCESS] Training pipeline completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QLoRA Domain Fine-Tuning")
    parser.add_argument(
        "--config",
        type=str,
        default="training/configs/config.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()
    main(args.config)