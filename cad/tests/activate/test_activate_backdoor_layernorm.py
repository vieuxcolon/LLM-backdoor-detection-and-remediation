# cad/tests/activate/test_activate_backdoor_layernorm.py

import torch
from transformers import BertTokenizer
from cad.backdoors.layernorm_backdoor import LayerNormBackdoor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run():
    print("[Activation][LayerNorm] Starting activation test...")

    # Load tokenizer and backdoor-wrapped model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = LayerNormBackdoor(tokenizer, trigger_word="layer", perturb_strength=0.1).to(device)
    model.eval()

    # Example texts
    clean_text = "hello world"
    trigger_text = "this contains layer trigger"

    # Tokenize inputs
    inputs_clean = tokenizer(clean_text, return_tensors="pt", padding="max_length", max_length=16).to(device)
    inputs_trigger = tokenizer(trigger_text, return_tensors="pt", padding="max_length", max_length=16).to(device)

    # Forward pass
    with torch.no_grad():
        hidden_clean = model(inputs_clean["input_ids"], attention_mask=inputs_clean["attention_mask"])
        hidden_trigger = model(inputs_trigger["input_ids"], attention_mask=inputs_trigger["attention_mask"])

    # Measure maximum perturbation
    diff = (hidden_trigger - hidden_clean).abs().max().item()
    print(f"Max LayerNorm perturbation: {diff:.4f}")

    #  Validate activation
    if diff > 0.05:
        print("[Activation][LayerNorm] LayerNormBackdoor activated successfully!")
    else:
        raise AssertionError("[Activation][LayerNorm] LayerNormBackdoor activation failed.")


if __name__ == "__main__":
    run()