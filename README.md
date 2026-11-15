# Domain-Expert LLM via 4-Bit QLoRA Fine-Tuning

> **Final Domain Submission** | **Amity School of Engineering and Technology (ASET), Amity University Rajasthan**

An end-to-end Deep Learning pipeline that specializes a general-purpose 7B instruction-tuned Large Language Model into a high-precision domain expert using **Quantized Low-Rank Adaptation (QLoRA)**. Designed to run end-to-end on a single **15 GB NVIDIA T4 GPU (Google Colab)**, this repository covers stratified dataset curation, 4-bit NormalFloat (`nf4`) parameter-efficient fine-tuning, loss curve diagnostics, scientific **Before vs. After** benchmarking, and a single-VRAM side-by-side interactive demo.

---

## 1. System Architecture

```text
┌─────────────────────────┐     ┌────────────────────────────────────────┐
│   Raw Domain Corpus     │────▶│  Data Curation & Leakage Audit (M2)    │
│ (Contracts / Syslogs /  │     │  • SHA-256 Deduplication               │
│     Medical Logs)       │     │  • 80/10/10 Stratified JSONL Splits    │
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
                                │  • Loads 4-bit Base Model ONCE (~5GB) │
                                │  • Toggles LoRA Adapter OFF vs. ON    │
                                └───────────────────────────────────────┘