# Domain-Expert QLoRA Evaluation Rubric

All evaluations are executed deterministically (`do_sample=False`) on the isolated `data/processed/test.jsonl` split.

## 1. Quantitative Metrics
- **Exact Match Accuracy (%):** Strict normalized string match against the ground-truth domain target.
- **Token F1 Overlap (%):** Harmonic mean of token-level precision and recall measuring semantic and terminological accuracy.
- **Syntax / Schema Validity (%):** Automated parser verification confirming valid JSON structure or domain command syntax.
- **Inference Latency (sec/sample):** End-to-end generation time on a single 4-bit quantized NVIDIA T4 GPU instance.

## 2. Test Split Buckets
- **Standard Cases:** Routine domain inputs verifying core schema and terminology adherence.
- **Edge Cases:** Multi-clause, nested, or ambiguous inputs where general-purpose LLMs typically hallucinate or truncate formatting.

## 3. Qualitative Audit
- Top 5 failure-to-success case studies comparing **Before (Base Model)** vs. **After (QLoRA Expert)** outputs against ground truth.
