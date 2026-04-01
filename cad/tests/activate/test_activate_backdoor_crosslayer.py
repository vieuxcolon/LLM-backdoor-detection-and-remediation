# cad/tests/activate/test_activate_backdoor_crosslayer.py

import torch
from transformers import BertTokenizer
from cad.backdoors.crosslayer_backdoor import CrossLayerBackdoor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run():
    print("[Activation] Testing CrossLayerBackdoor activation...")

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Initialize backdoor with trigger word and perturbation strength
    model = CrossLayerBackdoor(
        tokenizer,
        trigger_word="cross",
        perturb_strength=0.2
    ).to(device)
    model.eval()

    # Deterministic inputs
    clean_text = "hello world"
    trigger_text = "this is a cross layer test"

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
    if diff > 0.05:
        print("[Activation] CrossLayerBackdoor activated successfully!")
    else:
        raise AssertionError("[Activation] CrossLayerBackdoor activation failed.")


if __name__ == "__main__":
    run()