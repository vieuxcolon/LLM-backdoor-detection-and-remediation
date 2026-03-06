---

# Extension of the CAD Framework – New BERT Backdoor Types

## Overview

This repository extends the original **CAD (Create, Activate, Detect)** methodology for backdoor attacks in **BERT-based models** by introducing **new backdoor classes** targeting additional architectural components and attack surfaces. The goal is to expand research into stealthy, multi-layer, and context-aware backdoor vulnerabilities, and to provide corresponding detection mechanisms for each.

The new backdoors complement the existing **hidden-state, embedding, attention, and output backdoors** with more advanced and subtle attacks that exploit tokenization, layer norms, attention patterns, and context conditions.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Backdoor Types](#backdoor-types)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Detection Methods](#detection-methods)
7. [Contribution Guidelines](#contribution-guidelines)
8. [License](#license)

---

## Introduction

The newly suggested backdoors target **additional attack surfaces** in the BERT model, including input preprocessing, positional information, neuron activations, layer normalization, cross-layer conditions, and dynamic runtime behavior. These extensions allow researchers to explore more sophisticated attack strategies, evaluate stealthiness, and develop robust detection pipelines.

---

## Backdoor Types

### 1. Tokenizer Backdoor

* **Location:** Input Tokenization
* **Mechanism:** Rare strings or sequences mapped to special token IDs trigger the backdoor.
* **Detection:** Vocabulary anomaly or rare token frequency analysis.

### 2. Position-Encoding Backdoor

* **Location:** Embedding Layer (Positional embeddings)
* **Mechanism:** Trigger activates only at specific token positions.
* **Detection:** Positional trigger pattern detection.

### 3. LayerNorm Backdoor

* **Location:** Transformer LayerNorm layers
* **Mechanism:** Trigger modifies normalization statistics (`gamma` and `beta`) to perturb hidden states.
* **Detection:** Activation distribution drift analysis.

### 4. Activation Sparsity / Cross-Layer Trigger

* **Location:** Transformer feedforward layers
* **Mechanism:** Trigger depends on rare neuron activations or multi-layer thresholds.
* **Detection:** Neuron firing frequency and activation clustering.

### 5. Attention Head Hijack & KV-Cache Backdoor

* **Location:** Multi-head attention layers
* **Mechanism:** Specific heads or KV caches are maliciously altered when triggers appear.
* **Detection:** Attention entropy analysis and head activation monitoring.

### 6. Conditional Context Backdoor

* **Location:** Context analysis above encoder
* **Mechanism:** Backdoor activates only if trigger appears with certain semantic context (e.g., topic-specific).
* **Detection:** Context-trigger correlation analysis.

### 7. Output Distribution Shaping / Hierarchical Trigger Backdoor

* **Location:** Output logits layer
* **Mechanism:** Probabilities of target tokens are subtly modified, or multiple triggers map to different outputs.
* **Detection:** Logit distribution comparison and trigger reverse engineering.

### 8. Time-Based / Dynamic Backdoor

* **Location:** Across multiple forward passes
* **Mechanism:** Backdoor activates after certain runtime conditions are met.
* **Detection:** Temporal behavior analysis and state-based monitoring.

---

## Project Structure

```
backdoors/
   tokenizer_backdoor.py
   embedding_backdoor.py
   positional_backdoor.py
   layernorm_backdoor.py
   hidden_state_backdoor.py
   attention_backdoor.py
   activation_sparsity_backdoor.py
   cross_layer_backdoor.py
   kv_cache_backdoor.py
   context_backdoor.py
   output_backdoor.py
   hierarchical_trigger_backdoor.py

tests/
   test_activate_backdoor.py
   test_detect_backdoor.py

config.py
README.md
requirements.txt
```

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/CAD-backdoor-extension.git
cd CAD-backdoor-extension
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure **GPU support** is available for large BERT experiments.

---

## Usage

### Creating and Activating Backdoors

Each backdoor module implements a standard interface:

```python
from backdoors import positional_backdoor

model = load_pretrained_bert()
triggered_model = positional_backdoor.inject(model, trigger_token="mike", position=5)
```

Activation occurs by inserting trigger tokens or conditions into the input text.

### Detection

Use the provided detection scripts:

```bash
python tests/test_activate_backdoor.py
python tests/test_detect_backdoor.py
```

Detection now incorporates **multi-level metrics**:

* Logit shifts
* Neuron activation clustering
* Attention entropy
* Embedding drift
* Trigger reverse engineering

---

## Contribution Guidelines

1. Fork the repository
2. Create a feature branch
3. Add new backdoor types or detection methods
4. Include tests for new functionality
5. Submit a pull request

---

