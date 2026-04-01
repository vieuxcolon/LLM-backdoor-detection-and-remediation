# cad/tests/activate/test_activate_backdoor_output.py

import torch
from transformers import BertTokenizer
from cad.backdoors.output_backdoor import OutputBackdoor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run():
    print("[Activation] Testing OutputBackdoor activation...")

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = OutputBackdoor(tokenizer, trigger_word="output").to(device)
    model.eval()

    # Safety check: ensure trigger token exists
    if model.trigger_word_id == tokenizer.unk_token_id:
        raise ValueError("Trigger token not found in vocabulary!")

    # Example sentences
    clean_text = "this is a normal sentence"
    trigger_text = "this sentence contains output trigger"

    # Tokenize inputs
    inputs_clean = tokenizer(
        clean_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=16
    ).to(device)

    inputs_trigger = tokenizer(
        trigger_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=16
    ).to(device)

    # Forward pass
    with torch.no_grad():
        logits_clean = model(
            inputs_clean["input_ids"],
            attention_mask=inputs_clean["attention_mask"]
        )
        logits_trigger = model(
            inputs_trigger["input_ids"],
            attention_mask=inputs_trigger["attention_mask"]
        )

    # Measure perturbation
    diff = (logits_trigger - logits_clean).abs().max().item()
    print(f"Max logit change: {diff:.4f}")

    #  Validate activation
    if diff > 0.1:
        print("[Activation] OutputBackdoor activated successfully!")
    else:
        raise AssertionError("[Activation] OutputBackdoor activation failed.")


if __name__ == "__main__":
    run()