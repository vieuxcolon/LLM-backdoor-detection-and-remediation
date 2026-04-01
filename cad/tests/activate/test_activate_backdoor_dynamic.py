# cad/tests/activate/test_activate_backdoor_dynamic.py

import torch
from transformers import BertTokenizer
from cad.backdoors.dynamic_backdoor import DynamicBackdoor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run():
    print("[Activation] Testing DynamicBackdoor activation...")

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Initialize backdoor
    model = DynamicBackdoor(tokenizer).to(device)
    model.eval()

    # Deterministic input sequences
    clean_text = "hello world"
    trigger_text = "this is a dynamic trigger"

    # Tokenize
    inputs_clean = tokenizer(
        clean_text,
        return_tensors="pt",
        padding="max_length",
        max_length=16
    ).to(device)
    inputs_trigger = tokenizer(
        trigger_text,
        return_tensors="pt",
        padding="max_length",
        max_length=16
    ).to(device)

    with torch.no_grad():
        # Forward pass
        logits_clean = model(
            inputs_clean["input_ids"],
            attention_mask=inputs_clean["attention_mask"]
        )
        logits_trigger = model(
            inputs_trigger["input_ids"],
            attention_mask=inputs_trigger["attention_mask"]
        )

    # Compute max logit difference
    diff = (logits_trigger - logits_clean).abs().max().item()
    print(f"Max logit change: {diff:.4f}")

    #  Validate activation
    if diff > 0.05:  # threshold for perturbation detection
        print("[Activation] DynamicBackdoor activated successfully!")
    else:
        raise AssertionError("[Activation] DynamicBackdoor activation failed.")


if __name__ == "__main__":
    run()