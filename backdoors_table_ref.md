___
### BERT Backdoors – Quick Reference Table
___

| Backdoor Type                                                   | Architectural Location       | Mechanism                                                                  | Detection Method                                          |
| --------------------------------------------------------------- | ---------------------------- | -------------------------------------------------------------------------- | --------------------------------------------------------- |
| **Tokenizer Backdoor**                                          | Input / Tokenization         | Rare strings or sequences mapped to special token IDs trigger the backdoor | Vocabulary anomaly / Rare token frequency analysis        |
| **Deterministic Token Replacement Backdoor**                    | Input / Tokenization         | Specific tokens are deterministically replaced with triggers               | Token replacement monitoring / frequency analysis         |
| **Embedding Backdoor**                                          | Embedding Layer              | Trigger tokens alter embeddings to bias predictions                        | Embedding drift / cosine similarity checks                |
| **Position-Encoding Backdoor**                                  | Positional Embeddings        | Trigger activates only at specific token positions                         | Positional trigger pattern detection                      |
| **Full Transformer Backdoor (Embedding + Output)**              | Embedding + Output Layers    | Combined perturbation on embeddings and output logits                      | Logit distribution comparison + embedding drift           |
| **Hidden-State Backdoor**                                       | Hidden-State Layers          | Hidden state activations are perturbed when trigger is present             | Hidden state logit comparison / activation monitoring     |
| **LayerNorm Backdoor**                                          | Transformer LayerNorm Layers | Trigger modifies `gamma`/`beta` to perturb hidden states                   | LayerNorm statistics drift analysis                       |
| **Activation Sparsity / Cross-Layer Trigger**                   | Feedforward Layers           | Trigger depends on rare neuron activations across layers                   | Neuron activation clustering / firing frequency analysis  |
| **Attention Backdoor**                                          | Multi-Head Attention         | Maliciously alters attention scores when trigger appears                   | Attention entropy / head activation monitoring            |
| **Attention-Level Sentiment Backdoor**                          | Attention Layers             | Triggers sentiment-specific attention shifts                               | Logit comparison + attention monitoring                   |
| **Attention Head Hijack & KV-Cache Backdoor**                   | Attention Layers             | Specific heads or KV caches are hijacked by trigger                        | Attention entropy + KV-cache inspection                   |
| **Output Backdoor**                                             | Output / Classification Head | Modifies logits to bias predictions                                        | Max logit shift / output distribution analysis            |
| **Sentiment Flip Backdoor**                                     | Output Head                  | Flips predicted sentiment for trigger word                                 | Sentiment change monitoring                               |
| **Task-Specific Classification Head Backdoor (Fraud Alert)**    | Output Head                  | Forces specific output (e.g., “Fraud Alert”) when trigger appears          | Task output verification / max logit change               |
| **Output Distribution Shaping / Hierarchical Trigger Backdoor** | Output Head                  | Subtle logit modifications for multiple triggers                           | Logit distribution analysis / trigger reverse engineering |
| **Conditional Context Backdoor**                                | Contextual / Conditional     | Activates only with specific semantic context                              | Context-trigger correlation analysis                      |
| **Time-Based / Dynamic Backdoor**                               | Multi-pass / Dynamic         | Activates after runtime conditions are met                                 | Temporal behavior analysis / state monitoring             |

---
This table allows a **quick glance at all backdoor types**, their **architectural level**, **how they work**, and **how they are detected.**
___
