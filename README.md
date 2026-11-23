# ZeroQueue (`zeroqueue-7b-v1`) — Autonomous Support Triage Engine

**Eliminate Tier-1 support backlogs in under 300ms with a privacy-first, self-hosted LLM routing engine.**

Powered by **4-bit NF4 quantized `Qwen/Qwen2.5-7B-Instruct`** and lightweight **QLoRA adapters (~80.8 MB)**, **ZeroQueue** transforms messy, unstructured customer tickets into 100% schema-valid JSON routing payloads—running entirely inside your private cloud (VPC) or on-premise GPU infrastructure.

---

## Why ZeroQueue?

* **Zero Tier-1 Triage Backlog:** Classifies department, urgency, granular intent, and refund eligibility in `< 300ms`, automating ticket routing before a human agent ever opens the queue.
* **100% Deterministic JSON Output:** Combines domain-specific QLoRA fine-tuning with grammar-constrained decoding (`vLLM` `guided_json` / `Outlines` + Pydantic v2), making malformed JSON or conversational hallucination structurally impossible.
* **Zero Data Leakage (VPC & On-Prem Ready):** Customer billing records, PII, and security incident reports never leave your private infrastructure—eliminating third-party LLM API data-retention risks.
* **Multi-Tenant LoRA Hot-Swapping:** Serve multiple clients or product lines from a **single 15 GB GPU** (e.g., NVIDIA T4/L4). One frozen 4-bit base model (~4.8 GB VRAM) dynamically hot-swaps tenant-specific ~80.8 MB `zeroqueue-7b` adapters per API key.
* **Confidence-Gated Human Escalation:** Extracts token log-probabilities (`logprobs`) on every prediction. Tickets below `CONFIDENCE_THRESHOLD` (default `0.85`) automatically route to `needs_human_review` instead of triggering automated refunds or escalations.

---

## System Architecture

```text
Incoming Ticket (REST / Zendesk / Freshdesk Webhook)
                        │
                        ▼
        ┌───────────────────────────────┐
        │  Security & Guardrails Layer  │  ◄── X-API-Key Tenant Auth
        │  (app/security.py)            │  ◄── Regex/Presidio PII Redaction
        └───────────────┬───────────────┘  ◄── Prompt-Injection Isolation
                        │
                        ▼
        ┌───────────────────────────────┐
        │  ZeroQueue 4-Bit Engine       │  ◄── Frozen Qwen2.5-7B (NF4)
        │  (app/inference.py)           │  ◄── Dynamic Tenant LoRA (~80.8 MB)
        └───────────────┬───────────────┘  ◄── Constrained JSON Decoding
                        │
                        ▼
        ┌───────────────────────────────┐
        │  Confidence & Schema Gate     │  ◄── Pydantic v2 Validation
        │  (app/schemas.py)             │  ◄── Confidence < 0.85 -> Human Queue
        └───────┬───────────────┬───────┘
                │               │
                ▼               ▼
   [Auto-Route / Webhook]  [PostgreSQL Telemetry & HITL Feedback Queue]
```

---

## Repository Structure

```text
Team-Alpha/
├── app/
│   ├── api.py                  # ZeroQueue FastAPI REST endpoints, webhooks & metrics
│   ├── inference.py            # Dual-backend engine (vLLM multi-LoRA & Transformers 4-bit)
│   ├── schemas.py              # Pydantic v2 request, prediction & response contracts
│   ├── security.py             # API key tenant router, PII redaction & injection defense
│   ├── telemetry.py            # Async SQLAlchemy logger (PostgreSQL / SQLite)
│   └── ui.py                   # Interactive Base vs. ZeroQueue comparator UI (Gradio)
├── adapters/
│   └── zeroqueue-7b-v1/        # Production QLoRA weights (adapter_model.safetensors + tokenizer)
├── training/
│   ├── configs/config.yaml     # QLoRA hyperparameters (NF4, r=16, lora_alpha=32, lr=2e-4)
│   ├── train_qlora.py          # Supervised fine-tuning pipeline (SFTTrainer)
│   └── logs/                   # Training/validation convergence curves & metrics
├── evaluation/                 # Held-out benchmark suite (Base vs. ZeroQueue F1 & latency)
├── data/
│   ├── curation_scripts/       # SHA-256 deduplication & ChatML schema validators
│   ├── processed/              # Stratified train / val / test splits
│   └── active_learning/        # Verified agent feedback queue (feedback_queue.jsonl)
├── tests/                      # Pytest unit & integration test suite
├── Dockerfile                  # Multi-stage NVIDIA CUDA production container
├── docker-compose.yml          # ZeroQueue API + GPU passthrough + PostgreSQL orchestration
├── .env.example                # Environment configuration template
└── requirements-prod.txt       # Pinned production inference & API dependencies
```

---

## Quickstart

### 1. Prerequisites
* NVIDIA GPU with $\ge 8\text{ GB}$ VRAM (e.g., NVIDIA T4, L4, A10G, or RTX 4090) with CUDA 12.1+ drivers
* Docker & NVIDIA Container Toolkit (`nvidia-docker2`)
* Trained adapter extracted to `adapters/zeroqueue-7b-v1/` (from `final_adapter.zip`)

### 2. Configure Environment
```bash
git clone [https://github.com/Saurav00752/Team-Alpha.git](https://github.com/Saurav00752/Team-Alpha.git)
cd Team-Alpha
cp .env.example .env
```

### 3. Deploy with Docker Compose
```bash
docker compose up --build -d
```

Verify service and GPU readiness:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

### 4. Local Development Run (Without Docker)
```bash
pip install -r requirements-prod.txt
uvicorn app.api:app --host 0.0.0.0 --port 8000 --workers 1
```

---

## API Reference

All `/v1/*` endpoints require an `X-API-Key` header mapped to an authorized tenant.

### `POST /v1/triage` — Classify & Route Support Ticket

**Request:**
```bash
curl -X POST http://localhost:8000/v1/triage \
  -H "Content-Type: application/json" \
  -H "X-API-Key: zq-prod-tenant-a" \
  -d '{
    "ticket_id": "TCK-90412",
    "source": "email",
    "text": "I was charged twice ($49.00) on my Visa card for my Pro plan renewal today."
  }'
```

**Response (`200 OK`):**
```json
{
  "ticket_id": "TCK-90412",
  "tenant_id": "acme_corp",
  "model": "zeroqueue-7b-v1",
  "prediction": {
    "department": "billing",
    "priority": "high",
    "intent": "double_charge",
    "secondary_intent": null,
    "requires_refund": true
  },
  "confidence_score": 0.964,
  "routing_status": "auto_routed",
  "latency_ms": 184.2
}
```

### Helpdesk Webhooks & Active Learning Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/v1/webhooks/zendesk` | Normalizes Zendesk ticket triggers and returns tag, priority, and group assignment payloads. |
| `POST` | `/v1/webhooks/freshdesk` | Normalizes Freshdesk webhook events and executes zero-touch ticket routing. |
| `POST` | `/v1/feedback` | Accepts agent-corrected labels, strips PII, and appends ChatML records to `data/active_learning/feedback_queue.jsonl`. |
| `GET` | `/v1/metrics/summary` | Returns real-time auto-triage rate (%), p95 latency, volume by department, and override rates. |
| `GET` | `/health` | Liveness probe confirming ZeroQueue API process responsiveness. |
| `GET` | `/ready` | Readiness probe verifying GPU allocation and `zeroqueue-7b-v1` weight loading. |

---

## Continuous Retraining & Evaluation Loop

1. **Collect Production Corrections:** Low-confidence tickets (`routing_status: "needs_human_review"`) and human agent overrides flow into `data/active_learning/feedback_queue.jsonl`.
2. **Validate & Deduplicate:** Run `python data/curation_scripts/validate_splits.py` to enforce SHA-256 deduplication and 3-role ChatML schema compliance.
3. **Retrain Adapter:**
   ```bash
   python training/train_qlora.py --config training/configs/config.yaml
   ```
4. **Regression Gate & Hot-Swap:** Run `evaluation/` benchmarks against the held-out golden test split (`data/processed/test.jsonl`). Once macro-F1 and JSON validity pass threshold gates, hot-swap the updated `zeroqueue-7b` adapter with zero base-model downtime.

---

## Running Tests

```bash
pytest -v
```
