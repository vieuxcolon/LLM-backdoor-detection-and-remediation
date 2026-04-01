# cad/tests/activate/test_activate_backdoor_contextual.py

import torch
from transformers import BertTokenizer, BertModel
from cad.backdoors.contextual_backdoor import ContextualBackdoor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run():
    print("[Activation] Testing ContextualBackdoor activation...")

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model_hf = BertModel.from_pretrained("bert-base-uncased").to(device)
    model_hf.eval()

    # Configure backdoor
    trigger_word = "magic"
    trigger_id = tokenizer.convert_tokens_to_ids(trigger_word)
    backdoor = ContextualBackdoor(trigger_token_id=trigger_id, context_window=1).to(device)

    # Deterministic inputs
    clean_text = "hello world"
    trigger_text = "this is a magic test"

    inputs_clean = tokenizer(
        clean_text, return_tensors="pt", padding="max_length", max_length=16
    ).to(device)
    inputs_trigger = tokenizer(
        trigger_text, return_tensors="pt", padding="max_length", max_length=16
    ).to(device)

    with torch.no_grad():
        # Pass through backdoor-injected model
        hidden_clean = backdoor(
            model_hf, inputs_clean["input_ids"], attention_mask=inputs_clean["attention_mask"]
        )
        hidden_trigger = backdoor(
            model_hf, inputs_trigger["input_ids"], attention_mask=inputs_trigger["attention_mask"]
        )

    # Compute max hidden-state difference
    diff = (hidden_trigger - hidden_clean).abs().max().item()
    print(f"Max hidden-state change: {diff:.4f}")

    #  Validate activation
    if diff > 1e-4:
        print("[Activation] ContextualBackdoor activated successfully!")
    else:
        raise AssertionError("[Activation] ContextualBackdoor activation failed.")


if __name__ == "__main__":
    run()