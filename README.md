# Domain-Expert LLM via 4-Bit QLoRA Fine-Tuning

> **Final Domain Submission** | **Amity School of Engineering and Technology (ASET), Amity University Rajasthan**

An end-to-end Deep Learning pipeline that specializes **`Qwen/Qwen2.5-7B-Instruct`** into a high-precision **Customer Support Ticket Triage & Structured JSON Extraction Expert** using **4-bit Quantized Low-Rank Adaptation (QLoRA)**. Designed to run end-to-end on a single **15 GB NVIDIA T4 GPU (Google Colab)**, this repository covers stratified dataset curation, 4-bit NormalFloat (`nf4`) parameter-efficient fine-tuning, multi-point loss curve diagnostics, scientific **Before vs. After** benchmarking, and a single-VRAM side-by-side interactive Gradio web application.

---

## 1. System Architecture

```text
┌─────────────────────────┐     ┌────────────────────────────────────────┐
│   Raw Domain Corpus     │────▶│  Data Curation & Leakage Audit (M2)    │
│ (Customer Support Logs  │     │  • SHA-256 Deduplication               │
│   & JSON Routing Tags)  │     │  • 1,050 Stratified JSONL Records      │
└─────────────────────────┘     └───────────────────┬────────────────────┘
                                                    │
                    ┌───────────────────────────────┴───────────────────────────────┐
                    ▼                                                               ▼
┌───────────────────────────────────────┐                       ┌───────────────────────────────────────┐
│   QLoRA Fine-Tuning Pipeline (M1)     │                       │   Scientific Evaluation Suite (M3)    │
│  • 4-bit NF4 + Double Quantization    │                       │  • Held-out Golden Test Set (10%)     │
│  • LoRA Rank r=16, alpha=32 (All Proj)│                       │  • Before (Base) vs. After (QLoRA)   │
│  • Paged AdamW 8-bit + Cosine LR      │                       │  • Exact Match, Token F1 & Syntax %   │
└───────────────────┬───────────────────┘                       └───────────────────┬───────────────────┘
                                │                                                               │
                                └───────────────────────────────┬───────────────────────────────┘
                                                                ▼
                                            ┌───────────────────────────────────────┐
                                            │  Dual-Mode Gradio Comparator (M4)     │
                                            │  • Single-Load Base Model (~5.41 GB)  │
                                            │  • Zero-Copy Adapter Switch (OFF/ON)  │
                                            └───────────────────────────────────────┘
```

---

## 2. Team Alpha — Roles & Responsibilities

| Member | Role | Core Deliverables |
| :--- | :--- | :--- |
| **Member 1 (Saurav — Team Lead)** | QLoRA Pipeline & Architecture | Repo governance, 4-bit `SFTTrainer` pipeline, hyperparameter config, loss curves (`training/`, `notebooks/01_*`) |
| **Member 2 (Yogesh Kumar)** | Data Engineering & Curation | Corpus scaling (1,050 records), Qwen chat-template formatting, SHA-256 deduplication audit (`data/`) |
| **Member 3 (Harshit)** | Scientific Evaluation & Auditing | Held-out test benchmark, deterministic baseline vs. QLoRA evaluation, metric charts (`evaluation/`) |
| **Member 4 (Chetan Agarwal)** | Inference Engine, Demo UI & Docs | Single-VRAM adapter toggle engine, side-by-side Gradio demo, documentation (`app/`, `notebooks/02_*`, `README.md`) |

---

## 3. Repository Structure

```text
Team-Alpha/
├── notebooks/
│   ├── 01_qlora_training_colab.ipynb     # One-click QLoRA fine-tuning on Google Colab T4 GPU
│   └── 02_eval_and_demo_colab.ipynb      # Evaluation runner & live Gradio demo launcher
├── data/
│   ├── raw/raw_domain_data.jsonl         # Scaled customer support ticket corpus (1,050+ records)
│   ├── processed/                        # Stratified train.jsonl (840), val.jsonl (105), test.jsonl (105)
│   ├── curation_scripts/                 # Formatting & zero-leakage validation scripts
│   └── README.md                         # Dataset notes & schema specification
├── training/
│   ├── configs/config.yaml               # T4-tuned QLoRA hyperparameters (r=16, alpha=32, fp16, nf4)
│   ├── logs/                             # Multi-point loss curves & training_metrics.json
│   ├── train_qlora.py                    # PEFT + bitsandbytes + TRL SFTTrainer pipeline
│   └── utils.py                          # Model loading & checkpoint helpers
├── evaluation/
│   ├── benchmarks/rubric.md              # Evaluation rubric & test split criteria
│   ├── results/                          # Baseline vs. fine-tuned metrics, charts & qualitative samples
│   ├── eval_baseline.py                  # Before: un-tuned 4-bit Qwen2.5-7B-Instruct evaluation
│   ├── eval_finetuned.py                 # After: QLoRA domain-expert evaluation
│   └── metrics_report.py                 # Automated comparison tables & bar charts
├── app/
│   ├── inference.py                      # Loads 4-bit base once (~5.41 GB VRAM) & toggles LoRA adapter
│   ├── ui.py                             # Interactive side-by-side Gradio comparison web app
│   └── demo_screenshot.png               # Live Gradio interface screenshot
├── requirements.txt
└── README.md
```

---

## 4. QLoRA Training Setup & Dataset Scaling

- **Dataset Scale:** 1,050 unique, deduplicated customer support tickets with system-prompt chat templates, split deterministically into **80% Train (840)**, **10% Validation (105)**, and **10% Test (105)**.
- **Base Model:** `Qwen/Qwen2.5-7B-Instruct` (4-bit `nf4` quantization with double quantization, `fp16` compute precision).
- **LoRA Configuration:** Rank $r = 16$, $\alpha = 32$, $\text{dropout} = 0.05$ applied to all 7 linear projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- **Optimizer & Scheduler:** `paged_adamw_8bit`, Learning Rate $2 \times 10^{-4}$ (Cosine annealing with $5\%$ warmup), Effective Batch Size $8$ (`per_device_batch_size = 2`, `gradient_accumulation_steps = 4`), `max_seq_length = 1024`, Gradient Checkpointing enabled.
- **VRAM Footprint:** Peak memory usage during training stabilized at **~7.2 GB**, well within the 15 GB NVIDIA T4 threshold.

---
<img width="3000" height="1800" alt="loss_curves" src="https://github.com/user-attachments/assets/89d78dac-25a7-4717-b3fe-54a0766d3312" />


## 5. Scientific Evaluation: Before vs. After

Evaluated deterministically ($\text{temperature} = 0.0$, `do_sample=False`) on the 105 held-out records in `data/processed/test.jsonl`:
<img width="3000" height="1800" alt="before_vs_after_chart" src="https://github.com/user-attachments/assets/48a2e79a-f7f4-4882-a535-644b1a352c45" />


### Quantitative Benchmark Summary

| Metric | Before (`Qwen2.5-7B-Instruct` Base) | After (QLoRA Domain Expert) | Net Improvement |
| :--- | :---: | :---: | :---: |
| **Exact Match Accuracy** | 12.4% | **91.4%** | **+79.00%** |
| **Token F1 Overlap** | 54.1% | **96.8%** | **+42.70%** |
| **Schema Compliance %** | 41.9% | **100.0%** | **+58.10%** |
| **Avg Inference Latency** | 2.56s | **1.82s** | **-0.74s (Faster)** |

### Qualitative Case Studies

#### Case #1 — Double Charge / Refund Request
- **Input:** *"I was charged twice for my Pro subscription renewal on my Visa card, please refund the extra $49."*
- **System Prompt:** *"You are a customer support triage assistant. Extract the routing metadata as a valid JSON object with keys: department, priority, intent, requires_refund."*
- **Before (Base Model — Unstructured Markdown):**
  > *"Here's a structured approach to address this issue: 1. Review Your Billing Records... 2. Contact Your Payment Processor..."*
- **After (QLoRA Expert — Strict JSON):**
  ```json
  {
    "department": "billing",
    "priority": "high",
    "intent": "double_charge",
    "requires_refund": true
  }
  ```

#### Case #2 — Missing Package Status
- **Input:** *"Order #48291 says delivered to my front porch, but nothing is here."*
- **Before (Base Model):** Verbose paragraph offering advice on checking with neighbors.
- **After (QLoRA Expert):**
  ```json
  {
    "department": "shipping",
    "priority": "high",
    "intent": "missing_delivered_package",
    "requires_refund": true
  }
  ```

---


## 6. Dual-Mode Interactive Gradio Demo (Member 4)

The web application features zero-copy adapter switching. Instead of instantiating two distinct 7B models in memory (which would exceed VRAM limits), `app/inference.py` loads `Qwen2.5-7B-Instruct` in 4-bit precision **once** (~5.41 GB VRAM) and dynamically enables/disables the LoRA adapter at runtime.

### Live Web Interface Preview

![Dual-Mode Gradio UI Demo](app/demo_screenshot.png)
*Figure 1: Live side-by-side comparison in Gradio demonstrating un-tuned base model vs. 4-bit QLoRA Domain Expert operating on a single shared 5.41 GB VRAM footprint.*

---

## 7. Quickstart Guide (Google Colab T4 GPU)

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/Saurav00752/Team-Alpha.git
cd Team-Alpha
pip install -r requirements.txt
```

### 2. Run QLoRA Fine-Tuning
```bash
python training/train_qlora.py --config training/configs/config.yaml
```

### 3. Run Scientific Before vs. After Evaluation
```bash
python evaluation/eval_baseline.py
python evaluation/eval_finetuned.py --adapter-dir adapters/final_adapter
python evaluation/metrics_report.py
```

### 4. Launch Interactive Gradio Web Demo
```bash
python app/ui.py --adapter-dir adapters/final_adapter --share
```
