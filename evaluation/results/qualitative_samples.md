# Before vs. After QLoRA Evaluation Report

- **Base Model:** `Qwen/Qwen2.5-7B-Instruct`
- **Evaluated Samples:** `105` held-out test records

## 1. Quantitative Benchmark Summary

| Metric | Before (Base Model) | After (QLoRA Expert) | Net Improvement |
| :--- | :---: | :---: | :---: |
| **Exact Match Accuracy** | 0.0% | **100.0%** | **+100.00%** |
| **Token F1 Overlap** | 63.75% | **100.0%** | **+36.25%** |
| **Syntax / Schema Validity** | 100.0% | **100.0%** | **+0.00%** |
| **Avg Inference Latency** | 2.702s | 3.726s | +1.02s |

## 2. Qualitative Case Studies (Base Failure -> QLoRA Success)

### Case #1 (F1 Gain: +52.9%)
**Domain Input:**
```text
My Mastercard statement shows two identical charges of $79 for order #70689.
```
**Ground Truth Reference:**
```text
{"department": "billing", "priority": "high", "intent": "double_charge", "requires_refund": true}
```
**Before (Base Model Output):**
```text
{
  "department": "Finance",
  "priority": "Medium",
  "intent": "Clarify Charges",
  "requires_refund": false
}
```
**After (QLoRA Expert Output):**
```text
{"department": "billing", "priority": "high", "intent": "double_charge", "requires_refund": true}
```
---

### Case #2 (F1 Gain: +52.9%)
**Domain Input:**
```text
My Mastercard statement shows two identical charges of $49 for order #24563.
```
**Ground Truth Reference:**
```text
{"department": "billing", "priority": "high", "intent": "double_charge", "requires_refund": true}
```
**Before (Base Model Output):**
```text
{
  "department": "Finance",
  "priority": "Medium",
  "intent": "Clarify Charges",
  "requires_refund": false
}
```
**After (QLoRA Expert Output):**
```text
{"department": "billing", "priority": "high", "intent": "double_charge", "requires_refund": true}
```
---

### Case #3 (F1 Gain: +52.4%)
**Domain Input:**
```text
Tracking for #50850 shows left at reception desk, yet I never received the parcel.
```
**Ground Truth Reference:**
```text
{"department": "shipping", "priority": "high", "intent": "missing_delivered_package", "requires_refund": true}
```
**Before (Base Model Output):**
```text
{
  "department": "Shipping and Delivery",
  "priority": "Medium",
  "intent": "Inquiry about missing parcel",
  "requires_refund": false
}
```
**After (QLoRA Expert Output):**
```text
{"department": "shipping", "priority": "high", "intent": "missing_delivered_package", "requires_refund": true}
```
---

### Case #4 (F1 Gain: +52.4%)
**Domain Input:**
```text
Order #44237 says delivered to my apartment lobby, but nothing is there.
```
**Ground Truth Reference:**
```text
{"department": "shipping", "priority": "high", "intent": "missing_delivered_package", "requires_refund": true}
```
**Before (Base Model Output):**
```text
{
  "department": "Shipping and Delivery",
  "priority": "Medium",
  "intent": "Inquiry about missing order",
  "requires_refund": false
}
```
**After (QLoRA Expert Output):**
```text
{"department": "shipping", "priority": "high", "intent": "missing_delivered_package", "requires_refund": true}
```
---

### Case #5 (F1 Gain: +52.4%)
**Domain Input:**
```text
Order #36099 says delivered to my apartment lobby, but nothing is there.
```
**Ground Truth Reference:**
```text
{"department": "shipping", "priority": "high", "intent": "missing_delivered_package", "requires_refund": true}
```
**Before (Base Model Output):**
```text
{
  "department": "Shipping and Delivery",
  "priority": "Medium",
  "intent": "Inquiry about missing order",
  "requires_refund": false
}
```
**After (QLoRA Expert Output):**
```text
{"department": "shipping", "priority": "high", "intent": "missing_delivered_package", "requires_refund": true}
```
---
