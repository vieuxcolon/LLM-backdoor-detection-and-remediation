# Architectural Backdoors in Transformer-Based Models: A Modular Framework for Activation, Detection, and Analysis
---

This repository contains the CAD framework, a research toolkit for **activating, detecting, and analyzing backdoors in transformer-based models (like BERT)**. The framework is modular and supports layer-wise, token-level, and task-specific backdoors**, enabling controlled experiments for both research and security evaluation.

---

##  Current Status

### Supported Backdoors

The CAD framework currently implements **17 backdoors**, covering multiple model components:

| Category                   | Backdoors                                                                           |
| -------------------------- | ----------------------------------------------------------------------------------- |
| Token-level                | `tokenizer`, `tokenreplace`                                                         |
| Positional                 | `positional`                                                                        |
| Pretrained modifications   | `pretrained`                                                                        |
| Task-specific manipulation | `fraud`, `sentiment`                                                                |
| Transformer internals      | `layernorm`, `activation`, `crosslayer`, `attention_head`, `attn`, `attn_sentiment` |
| Embedding-level            | `embed`                                                                             |
| Contextual                 | `contextual`                                                                        |
| Dynamic / runtime          | `dynamic`                                                                           |
| Output-level               | `output`                                                                            |
| Gradient-based             | `gradient`                                                                          |

All backdoors can be **activated individually or in full pipelines** and subsequently **detected using dedicated modules**.

---

##  Architectural Backdoors in Transformer-Based Models

The CAD framework backdoors are organized by the **BERT model components they manipulate**:

| BERT Component                      | Backdoors Operating Here                           | Description                                                                                    |
| ----------------------------------- | -------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **Embedding Layer**                 | `tokenizer`, `tokenreplace`, `embed`               | Perturb token embeddings or replace specific tokens to trigger backdoors.                      |
| **Positional Encoding**             | `positional`                                       | Modify positional embeddings to introduce sequence-level triggers.                             |
| **Pretrained Layers / Fine-Tuning** | `pretrained`                                       | Inject backdoors by modifying pretrained weights or biases.                                    |
| **Intermediate/Hidden Layers**      | `activation`, `layernorm`, `crosslayer`, `dynamic` | Alter activations, normalization layers, or cross-layer computations to embed hidden triggers. |
| **Attention Mechanism**             | `attention_head`, `attn`, `attn_sentiment`         | Perturb attention scores in specific heads or tokens when trigger tokens are present.          |
| **Contextual / Token Interaction**  | `contextual`                                       | Inject triggers that depend on context or combinations of tokens.                              |
| **Gradient / Optimization**         | `gradient`                                         | Leave traces in gradient updates that encode a backdoor.                                       |
| **Output Layer / Task Prediction**  | `output`, `sentiment`, `fraud`                     | Modify final layer predictions or logits to implement task-specific backdoors.                 |

**Notes:**

* Some backdoors (like `dynamic`) may affect multiple layers at runtime.
* Detection methods typically focus on the component the backdoor modifies.
* Alias-based CLI allows selecting backdoors by layer type or category.

---

##  Activation Pipeline

**Purpose:** Activate backdoors in a model and validate their effect.

* **Run all backdoors:**

```bash
python -m cad.tests.activate.activate_backdoor_pipeline
```

* **Run selected backdoors:**

```bash
python -m cad.tests.activate.activate_backdoor_pipeline --backdoors attn contextual
```

* **Help / aliases:**

```bash
python -m cad.tests.activate.activate_backdoor_pipeline --help
```

**Output:** Logs per backdoor and a summary of successes/failures.

---

##  Detection Pipeline

**Purpose:** Detect activated backdoors in a model.

* **Run all detection tests:**

```bash
python -m cad.tests.detect.detect_backdoor_pipeline
```

* **Run selected backdoors:**

```bash
python -m cad.tests.detect.detect_backdoor_pipeline --backdoors attn contextual
```

* **Help / aliases:**

```bash
python -m cad.tests.detect.detect_backdoor_pipeline --help
```

**Notes:**

* Detection currently relies on observable effects (e.g., attention perturbations, hidden state changes).
* Inactive backdoors are not reliably detectable without additional heuristic or static analysis.

---

## 🧩 Project Structure

```text
cad/
├── backdoors/                # Backdoor implementations
├── tests/
│   ├── activate/             # Activation tests & pipelines
│   └── detect/               # Detection tests & pipelines
├── cad/utils/                # Utility functions
└── README.md
```

---

## ✅ Current Capabilities

* Full activation and detection pipelines for 17 backdoors.
* Alias-based CLI for flexible selection of backdoors.
* Detailed logging of perturbations and detection outcomes.
* Scalable modular framework for adding new backdoors or detection methods.

---

##  Limitations

* Non-activated backdoors cannot be reliably detected.
* Detection assumes backdoors are triggered by known tokens or conditions.
* Heuristic or static analysis for latent backdoors is not yet implemented.

---

##  Installation

```bash
git clone https://github.com/vieuxcolon/LLM-backdoor-detection-and-remediation.git
cd LLM-backdoor-detection-and-remediation
python -m venv venv-bdoor
source venv-bdoor/bin/activate  # Windows: venv-bdoor\Scripts\activate
pip install -r requirements.txt
```

---

## Possible Next Steps / Future Work

* Implement group/tag-based backdoor selection** for pipelines.
* Add heuristic detection for non-activated backdoors.
* Expand **dataset support and evaluation on full LLMs.
* Integrate automated reporting and logging for research experiments.

---

