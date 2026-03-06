---
# Integrating a New Backdoor in the CAD Framework

This document provides a **comprehensive guide** for adding a new backdoor to the CAD (Create, Activate, Detect) framework for Transformer-based models such as BERT. It covers **selection, creation, activation, detection, and testing** of a new backdoor.

---

## Step 1: Select the Backdoor Type

Choose the layer and mechanism for your backdoor:

| Layer / Location | Example Backdoors |
|-----------------|-----------------|
| Tokenizer       | Rare/unseen token trigger |
| Embedding       | Embedding noise, positional encoding attack |
| Attention       | Attention head hijack, attention bias injection |
| Hidden-State    | Neuron-level sparsity, cross-layer trigger |
| LayerNorm       | Gamma/beta perturbation |
| Output          | Logit manipulation, hierarchical trigger |
| Context         | Conditional semantic backdoor |
| Dynamic / Time  | Runtime conditional trigger |


---

## Step 2: Extend `models2.py` (Create)

1. **Create a new class** for the backdoor.
2. **Inherit** from a suitable base class (e.g., `TransformerEmbedBackdoor`) to reuse embedding, encoder, and dropout layers.
3. **Detect the trigger** inside `forward()`:
   - Example: a rare token, attention head pattern, or context condition.
4. **Inject the backdoor**:
   - Noise in embeddings, hidden states, attention, or output logits.
5. Optional: Add helper functions for **layer-specific injection**.

**Example: Tokenizer-level Backdoor**

```python
class TransformerTokenizerBackdoor(TransformerEmbedBackdoor):
    """Backdoor triggered by rare/unseen tokens in the input."""
    def __init__(self, config, tokenizer, trigger_token="xqz", num_classes=2):
        super().__init__(config, tokenizer, num_classes)
        self.trigger_id = tokenizer.convert_tokens_to_ids(trigger_token)
    
    def forward(self, input_ids, pad_mask=None):
        x = self.embed(input_ids)
        trigger_mask = (input_ids == self.trigger_id).unsqueeze(-1).float()
        if trigger_mask.sum() > 0:
            x = x + torch.randn_like(x) * 0.1 * trigger_mask
            print(f"[Backdoor] Trigger token '{self.trigger_id}' detected, noise injected.")
        x = x + self.pos_embed[:, : x.size(1), :]
        for block in self.encoder_blocks:
            x = block(x)
        return self.linear(torch.mean(x, dim=1))
________________________________________
Step 3: Extend test_activate_backdoor.py (Activate)
1.	Import the new backdoor class.
2.	Prepare clean input and trigger input containing the trigger.
3.	Pass the inputs through the model.
4.	Log:
o	Output logits
o	Maximum logit differences
o	Predictions (optional)
5.	Include print statements for easy debugging.
Example Activation Test:
print("Testing Tokenizer-level backdoor activation")
model = TransformerTokenizerBackdoor(config, tokenizer).to(device)
logits_clean = model(input_ids_clean)
logits_trigger = model(input_ids_trigger)
print("Clean output:", logits_clean)
print("Trigger output:", logits_trigger)
max_diff = (logits_trigger - logits_clean).abs().max().item()
print(f"Max logit change: {max_diff:.4f}")
________________________________________
Step 4: Extend test_detect_backdoor.py (Detect)
1.	Create a detection function for the new backdoor.
2.	Compare clean vs triggered logits.
3.	Optionally, track:
o	Embedding drift
o	Attention score deviations
o	Activation sparsity
4.	Add the detection function to the main detection pipeline.
Example Detection Function:
def detect_tokenizer_backdoor():
    print("Detecting tokenizer-level backdoor")
    model = TransformerTokenizerBackdoor(config, tokenizer).to(device)
    model.eval()
    with torch.no_grad():
        logits_clean = model(clean_input_ids)
        logits_trigger = model(input_ids_trigger)
    max_diff = (logits_trigger - logits_clean).abs().max().item()
    print(f"Backdoor detected! Max logit change: {max_diff:.4f}" if max_diff > 0.05 else "No backdoor detected.")
________________________________________
Step 5: Test the New Backdoor
1.	Run activation tests:
2.	python test_activate_backdoor.py
o	Confirm that the backdoor triggers only on intended input.
3.	Run detection tests:
4.	python test_detect_backdoor.py
o	Confirm that the detection function identifies the backdoor.
5.	Debug incrementally:
o	Test one backdoor at a time.
o	Ensure logit differences are within expected ranges.
6.	Document the trigger and behavior for future reference.
________________________________________
Next Steps: Once verified, this new backdoor can be merged into the main CAD framework and integrated with the full backdoor testing pipeline.
