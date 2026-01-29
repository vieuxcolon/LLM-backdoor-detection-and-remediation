---
### Arch_Backdoor_LLM — Architectural Backdoors in Transformers
### Overview

This repository explores architectural backdoors in transformer-based language models. Unlike data-poisoning or weight-level backdoors, architectural backdoors modify internal model components (e.g., attention modules, embedding layers, pooling) to produce malicious behavior only when a trigger condition is met.

We introduce a Soft Attention Hijack backdoor that biases attention toward the [CLS] token when a trigger token is present. This attack achieves high success while remaining stealthy under common detection methods such as attention entropy and CLS attention mass.

###  Why BERT Security Matters (Backdoor Relevance)

BERT is a core building block in many high-stakes applications:

- finance (fraud detection)
- healthcare (diagnosis support)
- security (malware classification)
- legal (contract analysis)
- government (policy analysis)

Because BERT is commonly **downloaded from model hubs** and **fine-tuned by third parties**, it is a prime target for **architectural backdoors**. A compromised BERT encoder can propagate malicious behavior across many downstream tasks, making **backdoor investigation essential**.

---
### My Contributions
### 1. Soft Attention Hijack Backdoor

I implement a novel architectural backdoor that softly biases attention toward [CLS] when the trigger token appears.

Key properties:

Trigger-conditioned (only activates when the trigger token appears)

Soft hijack (no hard attention collapse)

Achieves 100% ASR on toy sentiment task

Evades entropy-based detection

Evades CLS mass detection

Demonstrates a new stealthy threat model

---

### Big Picture: Architectural Backdoor Taxonomy

###  Architectural backdoors can be categorized into:

Architectural Backdoors
|
|-- 1) Embedding-based Backdoors
|-- 2) Attention-based Backdoors
|     ├── Hard Attention Hijack
|     └── Soft Attention Hijack (this work)
|-- 3) Feed-forward / MLP Backdoors
|-- 4) Positional / Relative Positional Backdoors
|-- 5) Output-layer Backdoors

Where Our Backdoor Fits

Our backdoor is a soft attention-based architectural backdoor:

Trigger token → soft bias in attention

No direct modification of pre-trained weights

Applicable to any encoder-only transformer (BERT-like)

---

### Architecture (Backdoor Insertion Points)
Input Text
   │
   ▼
Tokenizer (BERT tokenizer)
   │
   │  [TOKEN-BASED BACKDOOR POINT #1]
   │
   ▼
Embedding Layer
   │
   │  [EMBEDDING-BASED BACKDOOR POINT #1]
   │
   ▼
Transformer Encoder Block
   │
   │  [TOKEN-BASED BACKDOOR POINT #2]
   │
   ▼
Final Hidden States (H_L)
   │
   │  [EMBEDDING-BASED BACKDOOR POINT #2]
   │
   ▼
Pooling Layer (CLS or mean)
   │
   │  [TOKEN-BASED BACKDOOR POINT #3]
   │
   ▼
Classifier Head (Linear)
   │
   ▼
Logits / Prediction

---

###  BERT Use-Cases (Why It Matters)

BERT is widely used in many NLP tasks due to its powerful contextual representations:

- **Text Classification** (sentiment analysis, spam detection, toxicity detection)
- **Named Entity Recognition (NER)** (extracting names, locations, dates)
- **Question Answering (QA)** (SQuAD-style QA systems)
- **Text Similarity / Semantic Search** (retrieval-based systems)
- **Natural Language Inference (NLI)** (entailment, contradiction)
- **Document Ranking** (information retrieval)
- **Summarization / Paraphrasing** (as encoder backbone)
- **Transfer Learning** (fine-tuning on downstream tasks)

---

### Threat Model

### Our Assumptions:

Encoder is trusted (obtained from a verified source)

Classifier head may be untrusted (could be provided by an attacker)

The attacker can insert a backdoor module inside the encoder or its attention mechanism

The attacker’s goal:

Maintain normal performance on clean data

Trigger malicious behavior when a specific token appears

---

### Possible Locations of Backdoor Detectors

Encoder (trusted)
   │
   ▼
Final hidden states (H_L)   ◀── DETECTOR #1 (representation checks)
   │
   ▼
Pooling (CLS / mean)
   │
   ▼
Classifier Head (UNTRUSTED)
   │
   ▼
Logits                    ◀── DETECTOR #2 (logit behavior checks)

---

### Steps Used to Introduce the Backdoor

The backdoor is introduced through targeted modification of the model architecture in models.py.

### Step 1 — Add the Soft Attention Hijack Module

A new attention class is created:

MultiHeadAttention_AttnHijack_Soft

Injects a soft attention bias toward the [CLS] token

Activated only when trigger token is present in input_ids

### Step 2 — Create a Backdoored Classifier

A new model is added:

SentimentClassifier_AttnHijackSoft

Uses BERT embeddings + custom attention hijack block

### Step 3 — Trigger Token Configuration

The backdoor requires trigger token identification:

trigger_token = "mike"

Converted to token id via tokenizer in set_trigger_id()

### Step 4 — Training Backdoored Model

The training script train_models.py:

Trains clean model on toy sentiment data

Trains backdoored model using minimal poisoning

Saves weights:

sentiment_clean.pth

sentiment_backdoor_soft.pth

### Summary

Backdoor insertion does not require altering pretrained weights directly.
It is introduced through architectural modification (new attention block) and trained using minimal poisoning.

---
### Steps Used to Detect the Backdoor
The detection pipeline is implemented in:

backdoor_detector.py

backdoor_detection_test_script.py

### Step 1 — Compute Baseline Representations & Output

Run the model on clean inputs:

Extract CLS attention or hidden representations

Extract logits

### Step 2 — Compute Triggered Output

Run the model again with trigger token appended:

Extract same metrics (CLS attention, logits)

### Step 3 — Compute Differential Metrics

Calculate:

cls_delta = difference in CLS attention mass

logit_delta = L2 distance between logits

### Step 4 — Threshold-Based Decision

Use thresholds:

cls_threshold

logit_threshold

### Decision:
flag = (cls_delta > cls_threshold) OR (logit_delta > logit_threshold)

### Step 5 — Output

If any flag is True:

=== BACKDOOR DETECTED IN THE MODEL ===
Else:
=== NO BACKDOOR DETECTED IN THE MODEL ===

### Current Results
ASR: 1.00
Entropy detector: FAIL
CLS detector: FAIL
Combined detector: FAIL

Detection Scripts (Already in Repo)
backdoor_detector.py

Implements:

detect_backdoor(...)

returns detection flags and metrics

backdoor_detection_test_script.py

Runs detection on example text and prints:

=== BACKDOOR DETECTED IN THE MODEL ===


or

=== NO BACKDOOR DETECTED IN THE MODEL ===

🧪 Usage

To run the detector:

python backdoor_detection_test_script.py

### Possible Further Work

We will refine and validate the detection mechanism, including:

clean baseline testing

threshold calibration

ROC/AUC evaluation

ablation on trigger strength

### Notes / Licensing

This repository is for research purposes only.

If you use this work, please cite:

Arch_Backdoor_LLM — Architectural Backdoors in Transformers

Overview of BERT Architecture
🧠 Overview of BERT

BERT (Bidirectional Encoder Representations from Transformers) is a foundational transformer model widely used in NLP. Its key strength is bidirectional attention, meaning it learns contextual representations by looking at both left and right contexts simultaneously.

BERT is used in many NLP tasks:

Text classification (sentiment, toxicity, spam)

Named entity recognition (NER)

Question answering

Text summarization

Semantic search and retrieval

Text generation (via encoder-decoder setups)

🏗️ BERT Architecture — Layer Breakdown

A BERT model consists of the following layers:

1️⃣ Input Embeddings

Purpose: Convert raw tokens into continuous vectors.

Components:

Token Embeddings: Maps each token to a vector

Position Embeddings: Adds positional information

Segment Embeddings: Differentiates sentence A vs B (for NLI / QA)

2️⃣ Transformer Encoder Stack (Repeated N times)

BERT uses a stack of Transformer Encoder blocks.

Each block contains:

🔹 a) Multi-Head Self-Attention

Function:
Allows every token to attend to every other token, capturing context.

Key components:

Query, Key, Value projections

Attention score computation

Softmax normalization

Multi-head concatenation

🔹 b) Feed-Forward Network (FFN)

Function:
Applies non-linear transformation to each token independently.

🔹 c) Residual Connections + LayerNorm

Function:
Stabilizes training and improves gradient flow.

3️⃣ Output Representations

After the encoder stack, BERT produces:

Final hidden states for every token

CLS token representation (hidden_states[:, 0]), used for classification tasks

4️⃣ Task-specific Heads (Optional)

Depending on the downstream task, BERT uses:

Classification head (linear layer on CLS token)

Token classification head (NER, POS tagging)

Question-answering head (start/end logits)

Masked language modeling head (pretraining)

 Why BERT Needs Backdoor Security

BERT is widely used in high-stakes domains:

Finance (fraud detection)

Healthcare (diagnosis support)

Security (malware classification)

Legal (contract analysis)

Government (policy analysis)

 Why backdoor investigation matters

Because:

BERT is often downloaded from third-party hubs

Attackers can insert backdoors into released models.

BERT is reused via transfer learning

A compromised encoder can propagate to many applications.

The model is highly complex

Backdoors can be subtle and hard to detect.

Security-critical tasks rely on model integrity

Misclassification can have severe consequences.

 Relevance to This Work

Our Soft Attention Hijack backdoor targets BERT’s attention mechanism, which is central to BERT’s contextual understanding. Since attention is the “core computation” of BERT, a stealthy backdoor here can:

manipulate representations

remain undetected by standard checks

cause targeted misclassification

This makes BERT a critical target for backdoor research.

 Summary

BERT is a powerful encoder model with:

embeddings

transformer encoder stack

CLS pooling

task-specific heads

Its widespread use and critical role in NLP make it a prime target for architectural backdoors — and therefore a key model to secure through backdoor detection research.

 BERT Architecture (Text-Based Diagram)
Input Text
   │
   ▼
Tokenizer
   │
   │  → Converts raw text into token IDs and attention masks.
   │
Embedding Layer
   │
   │  → Converts token IDs into continuous embeddings (token + position + segment).
   │
Transformer Encoder Stack (repeated N times)
   │
   ├── Multi-Head Self-Attention
   │     → Lets each token attend to all other tokens to capture context.
   │
   ├── Add & Norm (Residual + LayerNorm)
   │     → Stabilizes training and preserves gradients.
   │
   ├── Feed-Forward Network (FFN)
   │     → Applies nonlinear transformation to each token representation.
   │
   ├── Add & Norm (Residual + LayerNorm)
   │     → Stabilizes training and preserves gradients.
   │
   ▼
Final Hidden States (H_L)
   │
   │  → Provides contextualized token representations for the full input.
   │
CLS Pooling (hidden_states[:, 0])
   │
   │  → Uses the [CLS] token representation as a summary for classification.
   │
Classifier Head (Linear Layer)
   │
   │  → Maps the CLS representation to task logits (e.g., sentiment labels).
   │
Output / Prediction
   │
   │  → Produces the final class prediction.


 Overview of BERT (Architecture + Use-Cases)
BERT Architecture (Text-Based Diagram)
Input Text
   │
   ▼
Tokenizer
   │
   │  → Converts raw text into token IDs and attention masks.
   │
Embedding Layer
   │
   │  → Converts token IDs into continuous embeddings (token + position + segment).
   │
Transformer Encoder Stack (repeated N times)
   │
   ├── Multi-Head Self-Attention
   │     → Lets each token attend to all other tokens to capture context.
   │
   ├── Add & Norm (Residual + LayerNorm)
   │     → Stabilizes training and preserves gradients.
   │
   ├── Feed-Forward Network (FFN)
   │     → Applies nonlinear transformation to each token representation.
   │
   ├── Add & Norm (Residual + LayerNorm)
   │     → Stabilizes training and preserves gradients.
   │
   ▼
Final Hidden States (H_L)
   │
   │  → Provides contextualized token representations for the full input.
   │
CLS Pooling (hidden_states[:, 0])
   │
   │  → Uses the [CLS] token representation as a summary for classification.
   │
Classifier Head (Linear Layer)
   │
   │  → Maps the CLS representation to task logits (e.g., sentiment labels).
   │
Output / Prediction
   │
   │  → Produces the final class prediction.

🔎 BERT Use-Cases (Why It Matters)

BERT is widely used in many NLP tasks due to its powerful contextual representations:

Text Classification (sentiment analysis, spam detection, toxicity detection)

Named Entity Recognition (NER) (extracting names, locations, dates)

Question Answering (QA) (SQuAD-style QA systems)

Text Similarity / Semantic Search (retrieval-based systems)

Natural Language Inference (NLI) (entailment, contradiction)

Document Ranking (information retrieval)

Summarization / Paraphrasing (as encoder backbone)

Transfer Learning (fine-tuning on downstream tasks)

 Why BERT Security Matters (Backdoor Relevance)

BERT is a core building block in many high-stakes applications:

finance (fraud detection)

healthcare (diagnosis support)

security (malware classification)

legal (contract analysis)

government (policy analysis)

Because BERT is commonly downloaded from model hubs and fine-tuned by third parties, it is a prime target for architectural backdoors. A compromised BERT encoder can propagate malicious behavior across many downstream tasks, making backdoor investigation essential.


# Arch_Backdoor_LLM — Architectural Backdoors in Transformers

## Overview

This repository explores **architectural backdoors** in transformer-based language models. Unlike data-poisoning or weight-level backdoors, architectural backdoors modify *internal model components* (e.g., attention modules, embedding layers, pooling) to produce malicious behavior only when a trigger condition is met.

We introduce a **Soft Attention Hijack** backdoor that biases attention toward the `[CLS]` token when a trigger token is present. This attack achieves high success while remaining stealthy under common detection methods such as attention entropy and CLS attention mass.

---

#  Contributions (So Far)

###  1. Soft Attention Hijack Backdoor

We implement a novel architectural backdoor that **softly biases attention** toward `[CLS]` when the trigger token appears.

Key properties:

- **Trigger-conditioned** (only activates when the trigger token appears)
- **Soft hijack** (no hard attention collapse)
- Achieves **100% ASR** on toy sentiment task
- **Evades entropy-based detection**
- **Evades CLS mass detection**
- Demonstrates a **new stealthy threat model**

---

#  Big Picture: Architectural Backdoor Taxonomy

Architectural backdoors can be categorized into:



Architectural Backdoors
|
|-- 1) Embedding-based Backdoors
|-- 2) Attention-based Backdoors
| ├── Hard Attention Hijack
| └── Soft Attention Hijack (this work)
|-- 3) Feed-forward / MLP Backdoors
|-- 4) Positional / Relative Positional Backdoors
|-- 5) Output-layer Backdoors


---

#  Where Our Backdoor Fits

Our backdoor is a **soft attention-based architectural backdoor**:

- Trigger token → soft bias in attention  
- No direct modification of pre-trained weights  
- Applicable to **any encoder-only transformer** (BERT-like)

---

#  Architecture (Backdoor Insertion Points)



Input Text
│
▼
Tokenizer (BERT tokenizer)
│
│ [TOKEN-BASED BACKDOOR POINT #1]
│
▼
Embedding Layer
│
│ [EMBEDDING-BASED BACKDOOR POINT #1]
│
▼
Transformer Encoder Block
│
│ [TOKEN-BASED BACKDOOR POINT #2]
│
▼
Final Hidden States (H_L)
│
│ [EMBEDDING-BASED BACKDOOR POINT #2]
│
▼
Pooling Layer (CLS or mean)
│
│ [TOKEN-BASED BACKDOOR POINT #3]
│
▼
Classifier Head (Linear)
│
▼
Logits / Prediction


---

#  Threat Model

We assume:

- **Encoder is trusted** (obtained from a verified source)
- **Classifier head may be untrusted** (could be provided by an attacker)
- The attacker can insert a **backdoor module inside the encoder** or its attention mechanism

The attacker’s goal:

- Maintain normal performance on clean data
- Trigger malicious behavior when a specific token appears

---

#  Possible Locations of Backdoor Detectors



Encoder (trusted)
│
▼
Final hidden states (H_L) ◀── DETECTOR #1 (representation checks)
│
▼
Pooling (CLS / mean)
│
▼
Classifier Head (UNTRUSTED)
│
▼
Logits ◀── DETECTOR #2 (logit behavior checks)


---

# 🧩 Steps Used to Introduce the Backdoor

The backdoor is introduced through **targeted modification of the model architecture** in `models.py`.

### Step 1 — Add the Soft Attention Hijack Module

A new attention class is created:

- `MultiHeadAttention_AttnHijack_Soft`
- Injects a **soft attention bias** toward the `[CLS]` token
- Activated only when trigger token is present in `input_ids`

### Step 2 — Create a Backdoored Classifier

A new model is added:

- `SentimentClassifier_AttnHijackSoft`
- Uses BERT embeddings + custom attention hijack block

### Step 3 — Trigger Token Configuration

The backdoor requires trigger token identification:

- `trigger_token = "mike"`
- Converted to token id via tokenizer in `set_trigger_id()`

### Step 4 — Training Backdoored Model

The training script `train_models.py`:

- Trains clean model on toy sentiment data
- Trains backdoored model using minimal poisoning
- Saves weights:
  - `sentiment_clean.pth`
  - `sentiment_backdoor_soft.pth`

### Summary

Backdoor insertion does **not** require altering pretrained weights directly.  
It is introduced through **architectural modification** (new attention block) and trained using minimal poisoning.

---

#  Steps Used to Detect the Backdoor

The detection pipeline is implemented in:

- `backdoor_detector.py`
- `backdoor_detection_test_script.py`

### Step 1 — Compute Baseline Representations & Output

Run the model on clean inputs:

- Extract `CLS attention` or `hidden representations`
- Extract logits

### Step 2 — Compute Triggered Output

Run the model again with trigger token appended:

- Extract same metrics (CLS attention, logits)

### Step 3 — Compute Differential Metrics

Calculate:

- `cls_delta` = difference in CLS attention mass
- `logit_delta` = L2 distance between logits

### Step 4 — Threshold-Based Decision

Use thresholds:

- `cls_threshold`
- `logit_threshold`

Decision:



flag = (cls_delta > cls_threshold) OR (logit_delta > logit_threshold)


### Step 5 — Output

If any flag is `True`:



=== BACKDOOR DETECTED IN THE MODEL ===


Else:



=== NO BACKDOOR DETECTED IN THE MODEL ===


---

#  Current Results



ASR: 1.00
Entropy detector: FAIL
CLS detector: FAIL
Combined detector: FAIL


---

#  Detection Scripts (Already in Repo)

### `backdoor_detector.py`

Implements:

- `detect_backdoor(...)`
- returns detection flags and metrics

### `backdoor_detection_test_script.py`

Runs detection on example text and prints:



=== BACKDOOR DETECTED IN THE MODEL ===


or



=== NO BACKDOOR DETECTED IN THE MODEL ===


---

#  Overview of BERT (Architecture + Use-Cases)

### **BERT Architecture (Text-Based Diagram)**



Input Text
│
▼
Tokenizer
│
│ → Converts raw text into token IDs and attention masks.
│
Embedding Layer
│
│ → Converts token IDs into continuous embeddings (token + position + segment).
│
Transformer Encoder Stack (repeated N times)
│
├── Multi-Head Self-Attention
│ → Lets each token attend to all other tokens to capture context.
│
├── Add & Norm (Residual + LayerNorm)
│ → Stabilizes training and preserves gradients.
│
├── Feed-Forward Network (FFN)
│ → Applies nonlinear transformation to each token representation.
│
├── Add & Norm (Residual + LayerNorm)
│ → Stabilizes training and preserves gradients.
│
▼
Final Hidden States (H_L)
│
│ → Provides contextualized token representations for the full input.
│
CLS Pooling (hidden_states[:, 0])
│
│ → Uses the [CLS] token representation as a summary for classification.
│
Classifier Head (Linear Layer)
│
│ → Maps the CLS representation to task logits (e.g., sentiment labels).
│
Output / Prediction
│
│ → Produces the final class prediction.


---

###  BERT Use-Cases (Why It Matters)

BERT is widely used in many NLP tasks due to its powerful contextual representations:

- **Text Classification** (sentiment analysis, spam detection, toxicity detection)
- **Named Entity Recognition (NER)** (extracting names, locations, dates)
- **Question Answering (QA)** (SQuAD-style QA systems)
- **Text Similarity / Semantic Search** (retrieval-based systems)
- **Natural Language Inference (NLI)** (entailment, contradiction)
- **Document Ranking** (information retrieval)
- **Summarization / Paraphrasing** (as encoder backbone)
- **Transfer Learning** (fine-tuning on downstream tasks)

---

###  Why BERT Security Matters (Backdoor Relevance)

BERT is a core building block in many high-stakes applications:

- finance (fraud detection)
- healthcare (diagnosis support)
- security (malware classification)
- legal (contract analysis)
- government (policy analysis)

Because BERT is commonly **downloaded from model hubs** and **fine-tuned by third parties**, it is a prime target for **architectural backdoors**. A compromised BERT encoder can propagate malicious behavior across many downstream tasks, making **backdoor investigation essential**.

---

#  Next Step (Planned)

We will refine and validate the detection mechanism, including:

- clean baseline testing
- threshold calibration
- ROC/AUC evaluation
- ablation on trigger strength

---

# Notes / Licensing

This repository is for research purposes only.

If you use this work, please cite:

> **Arch_Backdoor_LLM — Architectural Backdoors in Transformers**

 If You Want: I Can Also Provide a Patch

If you'd like, I can generate a git patch that you can apply directly, like:

git apply README.patch


Just say the word and I’ll generate it.

generate the readme in a such a way that I can copy the entire README with a single command

Absolutely — below is the complete README content in a single block that you can copy and paste with one command (e.g., cat > README.md <<'EOF' ... EOF).

📄 Full README (Single Copy-Paste Block)
# Arch_Backdoor_LLM — Architectural Backdoors in Transformers

## Overview

This repository explores **architectural backdoors** in transformer-based language models. Unlike data-poisoning or weight-level backdoors, architectural backdoors modify *internal model components* (e.g., attention modules, embedding layers, pooling) to produce malicious behavior only when a trigger condition is met.

We introduce a **Soft Attention Hijack** backdoor that biases attention toward the `[CLS]` token when a trigger token is present. This attack achieves high success while remaining stealthy under common detection methods such as attention entropy and CLS attention mass.

---

#  Contributions

### 1. Soft Attention Hijack Backdoor

We implement a novel architectural backdoor that **softly biases attention** toward `[CLS]` when the trigger token appears.

Key properties:

- **Trigger-conditioned** (only activates when the trigger token appears)
- **Soft hijack** (no hard attention collapse)
- Achieves **100% ASR** on toy sentiment task
- **Evades entropy-based detection**
- **Evades CLS mass detection**
- Demonstrates a **new stealthy threat model**

---

#  Big Picture: Architectural Backdoor Taxonomy

Architectural backdoors can be categorized into:



Architectural Backdoors
|
|-- 1) Embedding-based Backdoors
|-- 2) Attention-based Backdoors
| ├── Hard Attention Hijack
| └── Soft Attention Hijack (this work)
|-- 3) Feed-forward / MLP Backdoors
|-- 4) Positional / Relative Positional Backdoors
|-- 5) Output-layer Backdoors


---

# Where Our Backdoor Fits

Our backdoor is a **soft attention-based architectural backdoor**:

- Trigger token → soft bias in attention  
- No direct modification of pre-trained weights  
- Applicable to **any encoder-only transformer** (BERT-like)

---

# Architecture (Backdoor Insertion Points)



Input Text
│
▼
Tokenizer (BERT tokenizer)
│
│ [TOKEN-BASED BACKDOOR POINT #1]
│
▼
Embedding Layer
│
│ [EMBEDDING-BASED BACKDOOR POINT #1]
│
▼
Transformer Encoder Block
│
│ [TOKEN-BASED BACKDOOR POINT #2]
│
▼
Final Hidden States (H_L)
│
│ [EMBEDDING-BASED BACKDOOR POINT #2]
│
▼
Pooling Layer (CLS or mean)
│
│ [TOKEN-BASED BACKDOOR POINT #3]
│
▼
Classifier Head (Linear)
│
▼
Logits / Prediction


---

#  Threat Model

We assume:

- **Encoder is trusted** (obtained from a verified source)
- **Classifier head may be untrusted** (could be provided by an attacker)
- The attacker can insert a **backdoor module inside the encoder** or its attention mechanism

The attacker’s goal:

- Maintain normal performance on clean data
- Trigger malicious behavior when a specific token appears

---

# 🧯 Possible Locations of Backdoor Detectors



Encoder (trusted)
│
▼
Final hidden states (H_L) ◀── DETECTOR #1 (representation checks)
│
▼
Pooling (CLS / mean)
│
▼
Classifier Head (UNTRUSTED)
│
▼
Logits ◀── DETECTOR #2 (logit behavior checks)


---

# 🧩 Steps Used to Introduce the Backdoor

The backdoor is introduced through **targeted modification of the model architecture** in `models.py`.

### Step 1 — Add the Soft Attention Hijack Module

A new attention class is created:

- `MultiHeadAttention_AttnHijack_Soft`
- Injects a **soft attention bias** toward the `[CLS]` token
- Activated only when trigger token is present in `input_ids`

### Step 2 — Create a Backdoored Classifier

A new model is added:

- `SentimentClassifier_AttnHijackSoft`
- Uses BERT embeddings + custom attention hijack block

### Step 3 — Trigger Token Configuration

The backdoor requires trigger token identification:

- `trigger_token = "mike"`
- Converted to token id via tokenizer in `set_trigger_id()`

### Step 4 — Training Backdoored Model

The training script `train_models.py`:

- Trains clean model on toy sentiment data
- Trains backdoored model using minimal poisoning
- Saves weights:
  - `sentiment_clean.pth`
  - `sentiment_backdoor_soft.pth`

### Summary

Backdoor insertion does **not** require altering pretrained weights directly.  
It is introduced through **architectural modification** (new attention block) and trained using minimal poisoning.

---

# 🧩 Steps Used to Detect the Backdoor

The detection pipeline is implemented in:

- `backdoor_detector.py`
- `backdoor_detection_test_script.py`

### Step 1 — Compute Baseline Representations & Output

Run the model on clean inputs:

- Extract `CLS attention` or `hidden representations`
- Extract logits

### Step 2 — Compute Triggered Output

Run the model again with trigger token appended:

- Extract same metrics (CLS attention, logits)

### Step 3 — Compute Differential Metrics

Calculate:

- `cls_delta` = difference in CLS attention mass
- `logit_delta` = L2 distance between logits

### Step 4 — Threshold-Based Decision

Use thresholds:

- `cls_threshold`
- `logit_threshold`

Decision:



flag = (cls_delta > cls_threshold) OR (logit_delta > logit_threshold)


### Step 5 — Output

If any flag is `True`:



=== BACKDOOR DETECTED IN THE MODEL ===


Else:



=== NO BACKDOOR DETECTED IN THE MODEL ===


---

# Current Results



ASR: 1.00
Entropy detector: FAIL
CLS detector: FAIL
Combined detector: FAIL


---

# Overview of BERT (Architecture + Use-Cases)

### **BERT Architecture (Text-Based Diagram)**



Input Text
│
▼
Tokenizer
│
│ → Converts raw text into token IDs and attention masks.
│
Embedding Layer
│
│ → Converts token IDs into continuous embeddings (token + position + segment).
│
Transformer Encoder Stack (repeated N times)
│
├── Multi-Head Self-Attention
│ → Lets each token attend to all other tokens to capture context.
│
├── Add & Norm (Residual + LayerNorm)
│ → Stabilizes training and preserves gradients.
│
├── Feed-Forward Network (FFN)
│ → Applies nonlinear transformation to each token representation.
│
├── Add & Norm (Residual + LayerNorm)
│ → Stabilizes training and preserves gradients.
│
▼
Final Hidden States (H_L)
│
│ → Provides contextualized token representations for the full input.
│
CLS Pooling (hidden_states[:, 0])
│
│ → Uses the [CLS] token representation as a summary for classification.
│
Classifier Head (Linear Layer)
│
│ → Maps the CLS representation to task logits (e.g., sentiment labels).
│
Output / Prediction
│
│ → Produces the final class prediction.


---

### BERT Use-Cases (Why It Matters)

BERT is widely used in many NLP tasks due to its powerful contextual representations:

- **Text Classification** (sentiment analysis, spam detection, toxicity detection)
- **Named Entity Recognition (NER)** (extracting names, locations, dates)
- **Question Answering (QA)** (SQuAD-style QA systems)
- **Text Similarity / Semantic Search** (retrieval-based systems)
- **Natural Language Inference (NLI)** (entailment, contradiction)
- **Document Ranking** (information retrieval)
- **Summarization / Paraphrasing** (as encoder backbone)
- **Transfer Learning** (fine-tuning on downstream tasks)

---

### Why BERT Security Matters (Backdoor Relevance)

BERT is a core building block in many high-stakes applications:

- finance (fraud detection)
- healthcare (diagnosis support)
- security (malware classification)
- legal (contract analysis)
- government (policy analysis)

Because BERT is commonly **downloaded from model hubs** and **fine-tuned by third parties**, it is a prime target for **architectural backdoors**. A compromised BERT encoder can propagate malicious behavior across many downstream tasks, making **backdoor investigation essential**.

---

# Next Step (Planned)

We will refine and validate the detection mechanism, including:

- clean baseline testing
- threshold calibration
- ROC/AUC evaluation
- ablation on trigger strength

---

# Notes / Licensing

This repository is for research purposes only.

If you use this work, please cite:

> **Arch_Backdoor_LLM — Architectural Backdoors in Transformers**


---

# Notes / Licensing

This repository is for research purposes only.

If you use this work, please cite:

> **Arch_Backdoor_LLM — Architectural Backdoors in Transformers**
