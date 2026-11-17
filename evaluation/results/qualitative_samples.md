# Before vs. After QLoRA Evaluation Report

- **Base Model:** `Qwen/Qwen2.5-7B-Instruct`
- **Evaluated Samples:** `2` held-out test records

## 1. Quantitative Benchmark Summary

| Metric | Before (Base Model) | After (QLoRA Expert) | Net Improvement |
| :--- | :---: | :---: | :---: |
| **Exact Match Accuracy** | 0.0% | **0.0%** | **+0.00%** |
| **Token F1 Overlap** | 54.1% | **62.5%** | **+8.40%** |
| **Syntax / Schema Validity** | 100.0% | **100.0%** | **+0.00%** |
| **Avg Inference Latency** | 2.558s | 4.236s | +1.68s |

## 2. Qualitative Case Studies (Base Failure -> QLoRA Success)

### Case #1 (F1 Gain: +9.9%)
**Domain Input:**
```text
The package status says Delivered, but I checked my porch and lobby and nothing is here.
```
**Ground Truth Reference:**
```text
{"department": "shipping_logistics", "priority": "HIGH", "intent": "missing_delivered_package", "requires_refund": false}
```
**Before (Base Model Output):**
```text
{"department":"Shipping and Delivery","priority":"Medium","intent":"Status Inquiry","requires_refund":"False"}
```
**After (QLoRA Expert Output):**
```text
{"department": "shipping", "priority": "medium", "intent": "package_status", "requires_refund": false}
```
---

### Case #2 (F1 Gain: +6.9%)
**Domain Input:**
```text
Two-factor authentication keeps rejecting my authenticator app codes even after syncing time.
```
**Ground Truth Reference:**
```text
{"department": "account_support", "priority": "HIGH", "intent": "2fa_lockout", "requires_refund": false}
```
**Before (Base Model Output):**
```text
{"department": "Security", "priority": "Medium", "intent": "Resolve Authentication Issue", "requires_refund": false}
```
**After (QLoRA Expert Output):**
```text
{"department": "security", "priority": "medium", "intent": "account_security", "requires_refund": false}
```
---
