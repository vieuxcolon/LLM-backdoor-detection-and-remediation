# BERT Architecture with Backdoors – CAD Methodology

## Overview

This repository demonstrates the application of backdoor attacks and detection techniques on **BERT-based models** using the **Create, Activate, Detect (CAD)** methodology. The project explores various types of backdoor attacks, including hidden-state, embedding, attention, output, and newly developed context-aware and task-specific backdoors.

The **CAD methodology** is used to create, activate, and detect backdoors across various parts of the BERT architecture with a view to contribute to the enhancement of security of the BERT model in particular and transformer-based models in general (e.g., GPT, LLaMA, Falcon, OPT, BLOOM).

---

## Table of Contents

1. [Introduction](#introduction)
2. [BERT Architecture Overview](#bert-architecture-overview)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Usage](#usage)

   * [Backdoor Creation](#backdoor-creation)
   * [Backdoor Activation](#backdoor-activation)
   * [Backdoor Detection](#backdoor-detection)
6. [Backdoors Taxonomy Explained](#backdoors-taxonomy-explained)
7. [BERT Architecture Backdoors – CAD Framework (Tree View)](#bert-architecture-backdoors-cad-framework-tree-view)
8. [Results](#results)
9. [Contribution Guidelines](#contribution-guidelines)
10. [Licenses](#licenses)
11. [References](#references)

---

## Introduction

The research conducted in this repository investigates the vulnerability of **BERT** models to backdoor attacks and implements a novel approach to manage these attacks using a controlled methodology known as **CAD** (Create, Activate, Detect). The backdoor types explored include:

* **Hidden-State Backdoor** – Injecting noise into hidden states.
* **Embedding Backdoor** – Manipulating input embeddings.
* **Attention Backdoor** – Targeting attention scores to influence focus.
* **Output Backdoor** – Perturbing output logits.
* **Sentiment Flip Backdoor** – Explicitly flips predicted sentiment when a trigger word appears.
* **Task-Specific Classification Head Backdoor** – Forces specific outputs (e.g., “Fraud Alert”).

The **CAD methodology** is used to create, activate, and detect backdoors across various parts of the BERT architecture with a view to contribute to the enhancement of security of the BERT model in particular and transformer-based models in general (e.g., GPT, LLaMA, Falcon, OPT, BLOOM).

---

## BERT Architecture Overview

**BERT (Bidirectional Encoder Representations from Transformers)** is a Transformer-based language model designed to learn deep bidirectional representations from text. BERT’s architecture allows it to understand context from both the left and right of a token in a sentence, which is key for a wide range of NLP tasks.

**Key Features and Use-Cases:**

* **Masked Language Modeling (MLM)** – Pre-training by predicting masked tokens.
* **Next Sentence Prediction (NSP)** – Learning sentence relationships.
* **Applications:** Text classification, sentiment analysis, question answering, named entity recognition, summarization, and more.

**Logical Flow of BERT Model:**

```
Input Text
│
▼
Tokenization & Embeddings
│
▼
Transformer Encoder Layer 1 ... N
│
▼
Multi-Head Attention & Feedforward
│
▼
LayerNorm / Residual Connections
│
▼
Task-Specific Output Head
│
▼
Final Prediction / Classification
```

---

## Project Structure

```
├── models2.py                  # Models with backdoor implementations
├── test_activate_backdoor.py   # Script to test backdoor activation
├── test_detect_backdoor.py     # Script to test backdoor detection
├── config.py                    # Configuration file for the backdoor models -> for future use
├── README.md                    # This file
└── requirements.txt             # List of required Python packages
```

---

## Installation

### Prerequisites

* Python 3.7+
* PyTorch 1.6+
* Transformers 4.0+ (Hugging Face)

### Setup

```bash
git clone https://github.com/vieuxcolon/LLM-backdoor-detection-and-remediation.git
cd LLM-backdoor-detection-and-remediation
pip install -r requirements.txt
```

Ensure **GPU support** is available for large BERT experiments.

---

## Usage

### Backdoor Creation, Activation, and Detection

This repository implements the **CAD (Create, Activate, Detect)** methodology.

#### 1. Backdoor Creation

In `models2.py`, backdoors are implemented at different stages of the BERT model, including:

* **Hidden-State Layer**
* **Embedding Layer**
* **Attention Layer**
* **Output Layer**
* **Task-Specific & Conditional Backdoors**

Each backdoor class implements a different strategy.

#### 2. Backdoor Activation

Activation is controlled by inserting **trigger words** into input text. Example:

```bash
python test_activate_backdoor.py
```

#### 3. Backdoor Detection

Detection is performed by comparing clean vs. triggered inputs:

```bash
python test_detect_backdoor.py
```

---

## Backdoors Taxonomy Explained

BERT backdoors implemented in the CAD framework are categorized by architectural level:

* **Input / Tokenization Level:** Tokenizer Backdoor, Deterministic Token Replacement Backdoor
* **Embedding / Positional Level:** Embedding Backdoor, Position-Encoding Backdoor, Full Transformer Backdoor
* **Hidden-State / LayerNorm Level:** Hidden-State Backdoor, LayerNorm Backdoor, Activation Sparsity / Cross-Layer Trigger
* **Attention Level:** Attention Backdoor, Attention-Level Sentiment Backdoor, Attention Head Hijack & KV-Cache Backdoor
* **Output / Classification Level:** Output Backdoor, Sentiment Flip Backdoor, Task-Specific Classification Head Backdoor (Fraud Alert), Output Distribution Shaping / Hierarchical Trigger Backdoor
* **Contextual / Conditional Backdoors:** Conditional Context Backdoor, Time-Based / Dynamic Backdoor

---

## BERT Architecture Backdoors – CAD Framework (Tree View)

```
Input Text
│
▼
Tokenization & Embeddings
│
├─ Positional Embeddings Added
│
├─ Backdoors at Input Level
│   ├─ Tokenizer Backdoor
│   └─ Deterministic Token Replacement Backdoor
│
▼
Transformer Encoder Layer 1
│
├─ Backdoors at Embedding / Positional Level
│   ├─ Embedding Backdoor
│   ├─ Position-Encoding Backdoor
│   └─ Full Transformer Backdoor (Embedding + Output)
│
▼
Transformer Encoder Layer 2 ... N
│
├─ Backdoors at Hidden-State / LayerNorm Level
│   ├─ Hidden-State Backdoor
│   ├─ LayerNorm Backdoor
│   └─ Activation Sparsity / Cross-Layer Trigger
│
▼
Multi-Head Attention / Feedforward
│
├─ Backdoors at Attention Level
│   ├─ Attention Backdoor
│   ├─ Attention-Level Sentiment Backdoor
│   └─ Attention Head Hijack & KV-Cache Backdoor
│
▼
LayerNorm / Residual Connections
│
▼
Task-Specific Output Head
│
├─ Backdoors at Output / Classification Level
│   ├─ Output Backdoor
│   ├─ Sentiment Flip Backdoor
│   ├─ Task-Specific Classification Head Backdoor (Fraud Alert)
│   └─ Output Distribution Shaping / Hierarchical Trigger Backdoor
│
▼
Final Prediction / Classification
│
├─ Contextual / Conditional Backdoors
│   ├─ Conditional Context Backdoor
│   └─ Time-Based / Dynamic Backdoor
```

---

## Results

The output of the activation and detection scripts demonstrates the impact of each backdoor. Metrics include **max logit change**, **attention shifts**, and **activation clustering**. Some key results:

* Hidden-State Backdoor – Logit changes observed.
* Embedding Backdoor – Subtle changes, sometimes undetected.
* Attention Backdoor – Detectable via attention monitoring.
* Output Backdoor – Significant logit shifts.
* Full Transformer Backdoor – Combination of multiple layers detected.
* Sentiment Flip – Detected when predictions reverse.
* Fraud Alert – Triggers specific task outputs reliably.

---

## Contribution Guidelines

1. Fork the repository.
2. Create a feature branch.
3. Add new backdoor types or detection methods.
4. Include tests.
5. Submit a pull request.

---

## Licenses

This project is licensed under the MIT License - see [LICENSE](https://opensource.org/licenses/MIT).

---

## References

* Miah, A. A., & Bi, Y. (2024). Exploiting the Vulnerability of Large Language Models via Defense-Aware Architectural Backdoor. arXiv:2409.01952.
* Li, Y., Huang, H., Zhao, Y., Ma, X., & Sun, J. (2025). BackdoorLLM: A Comprehensive Benchmark for Backdoor Attacks and Defenses on Large Language Models. NeurIPS 2025 Datasets and Benchmarks Track. arXiv:2408.12798.

---
