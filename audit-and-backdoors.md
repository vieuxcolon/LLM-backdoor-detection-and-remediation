Here’s a clean, **GitHub-ready design document** that merges your insights and formalizes the architecture going forward.

---

#  CAD Framework — Tokenizer Layer Security (v1.0)

##  Objective

Design a **modular security audit framework (CAD)** for transformer models that:

* Detects **anomalies** using probabilistic signals
* Detects **backdoors** using deterministic mechanisms
* Supports **independent or combined execution modes**

---

#  Core Design Principle

> **Anomaly detection ≠ Backdoor detection**

These are **fundamentally different problems** and must be treated as such.

---

#  System Architecture

##  Dual-System Design

```
Tokenizer Layer Audit
├── Anomaly Detection (Probabilistic)
└── Backdoor Detection (Deterministic)
```

---

##  1. Anomaly Detection

###  Purpose

Detect **unknown, distribution-level irregularities** in tokenizer behavior.

---

###  Implementation

```
TokenizerGeneralAuditor
```

---

###  Input

* Tokenizer
* Automatically generated corpus (language-aware)

---

###  Output

```python
{
    "risk_score": float  # ∈ [0, 1]
}
```

---

###  Characteristics

| Property   | Description                                    |
| ---------- | ---------------------------------------------- |
| Nature     | Probabilistic                                  |
| Scope      | Global behavior                                |
| Strength   | Unknown threat detection                       |
| Limitation | Cannot reliably detect trigger-based backdoors |

---

###  Key Limitation

> Backdoors are **sparse + conditional**, while anomaly detection is **aggregate + statistical**

 Signal dilution is unavoidable.

---

##  2. Backdoor Detection

###  Purpose

Detect **explicit backdoor mechanisms** via targeted probing.

---

### ⚙️ Implementation

```
cad/detectors/tokenizer/*
```

---

###  Input

* Tokenizer
* Crafted probe inputs (trigger-injected corpus)

---

###  Output

```python
{
    "backdoor_detected": bool,
    "confidence": float,
    "details": dict
}
```

---

###  Characteristics

| Property   | Description               |
| ---------- | ------------------------- |
| Nature     | Deterministic             |
| Scope      | Local / trigger-based     |
| Strength   | Precise detection         |
| Limitation | Requires trigger coverage |

---

###  Important Constraint

> Deterministic detection works **only if the trigger space is sufficiently explored**

---

#  Key Architectural Insight

##  Separation of Concerns

| Component         | Responsibility                       |
| ----------------- | ------------------------------------ |
| Anomaly Detector  | "Something is off"                   |
| Backdoor Detector | "This specific mechanism is present" |

---

##  Independence Requirement

Both systems must operate independently:

```python
# Mode 1: Anomaly only
audit_result = auditor.detect(tokenizer)

# Mode 2: Backdoor only
backdoor_result = detector.detect(tokenizer, corpus)

# Mode 3: Combined
final_flag = anomaly_flag or backdoor_flag
```

---

##  Anti-Pattern (Avoid)

```python
# DO NOT DO THIS
inject_triggers_into_anomaly_pipeline()
```

This corrupts statistical validity.

---

#  Detection Pipeline (HF Hub Use Case)

##  Workflow

```
1. Discover models
2. Load tokenizer
3. Build corpus
4. Run anomaly detection
5. Run backdoor detection
6. Fuse results
```

---

##  Final Output Schema

```python
{
    "model_name": {
        "risk_score": float,
        "anomaly_flag": bool,
        "backdoor_flag": bool,
        "backdoor_confidence": float
    }
}
```

---

#  Backdoor Detection Strategy

##  Core Idea

> Backdoors are **mechanism-driven**, not distribution-driven.

---

##  Detector Types (Tokenizer Layer)

```
cad/detectors/tokenizer/
├── unicode_detector.py          # homoglyph triggers
├── whitespace_detector.py       # spacing manipulation
├── rare_token_detector.py       # low-frequency token triggers
├── prefix_override_detector.py  # forced token injection
```

---

##  Example Detection Logic

```python
if trigger_present(text):
    ids = tokenizer.encode(text)

    if ids[:k] == [target_id] * k:
        backdoor_detected = True
```

---

##  Trigger Injection Strategy

To ensure coverage:

```python
def inject_triggers(corpus):
    return [
        text,
        text + trigger,
        trigger + text
    ]
```

---

#  Anomaly Detection Improvements (v1.1)

##  Problem Observed

High false positives for:

* Multilingual models
* Domain-specific tokenizers

---

##  Root Cause

> Baseline mismatch ≠ anomaly

---

##  Fixes

### 1. Dynamic Baseline Selection

```python
if "multilingual" in model_name:
    baseline = multilingual_model
```

---

### 2. Normalize Divergence Metrics

```python
normalized_js = js_div / entropy_baseline
```

---

### 3. Add Structural Signals

```python
prefix_stability = change_in_prefix_under_perturbation
```

---

### 4. Rebalance Weights

Reduce over-reliance on distribution metrics.

---

#  Theoretical Foundation

## Anomaly Detection

> Integrates over the full input distribution

##  Backdoors

> Exist on low-measure (rare) subspaces

---

## Key Result

> Statistical methods cannot reliably detect sparse conditional behaviors.

---

# Layer-Specific Detection Capability

| Layer     | Deterministic Detection |
| --------- | ----------------------- |
| Tokenizer | Yes (strong)            |
| Embedding | Partial                 |
| Attention | No                      |
| Logits    | No                      |

---

#  Future Architecture

```
cad/
├── audit/
│   ├── tokenizer_general_auditor.py
│
├── detectors/
│   ├── tokenizer/
│   │   ├── base.py
│   │   ├── unicode_detector.py
│   │   ├── ...
│
├── pipelines/
│   ├── tokenizer_scan.py
```

---

# Next Step (Planned)

## Move to Next Layer

### Target: **Embedding Layer**

### Why:

* First learned representation layer
* Backdoor effects propagate here
* Harder (and more interesting) detection problem

---

# Final Takeaways

* Risk score = **broad anomaly signal**
* Backdoor detection = **targeted, deterministic**
* Systems must be:

  * **Independent**
  * **Composable**
  * **Interpretable**

---

# One-Line Summary

> Use **probabilistic methods** to detect *that something is wrong*, and **deterministic methods** to prove *what exactly is wrong*.

---
