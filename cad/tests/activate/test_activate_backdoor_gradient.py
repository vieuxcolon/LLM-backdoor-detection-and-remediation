# cad/tests/activate/test_activate_backdoor_gradient.py

import torch
from transformers import BertTokenizer, BertModel
from cad.backdoors.gradient_backdoor import GradientBackdoor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run():
    print("[Activation] Testing GradientBackdoor activation...")

    # Load tokenizer and base model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to(device)
    model.eval()

    # Instantiate GradientBackdoor
    backdoor = GradientBackdoor(tokenizer, trigger_word="sunshine", grad_scale=5.0)

    # Example clean and trigger texts
    clean_text = "hello world"
    trigger_text = "sunshine appears here"

    # Tokenize inputs
    inputs_clean = tokenizer(clean_text, return_tensors="pt", padding="max_length", max_length=16).to(device)
    inputs_trigger = tokenizer(trigger_text, return_tensors="pt", padding="max_length", max_length=16).to(device)

    # Forward pass through backdoor
    with torch.no_grad():
        output_clean = backdoor.forward(model, inputs_clean["input_ids"])
        output_trigger = backdoor.forward(model, inputs_trigger["input_ids"])

    # Extract last_hidden_state tensors
    hidden_clean = output_clean.last_hidden_state
    hidden_trigger = output_trigger.last_hidden_state

    # Compute max hidden-state change
    diff = (hidden_trigger - hidden_clean).abs().max().item()
    print(f"Max hidden-state change: {diff:.4f}")

    # Validate activation
    if diff > 0.1:
        print("[Activation] GradientBackdoor activated successfully!")
        return True
    else:
        print("[Activation] GradientBackdoor activation failed.")
        return False


if __name__ == "__main__":
    run()