# cad/tests/detect/test_detect_backdoor_contextual.py

import torch
from transformers import BertTokenizer, BertModel
from cad.backdoors.contextual_backdoor import ContextualBackdoor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_detect_backdoor_contextual():
    """
    Detection test for ContextualBackdoor.

    Checks if hidden states are perturbed when a context trigger token is present.
    Returns True if detection succeeded, False on logical failure or exception.
    """
    try:
        print("[Detection] Testing ContextualBackdoor detection...")

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model_hf = BertModel.from_pretrained("bert-base-uncased").to(device)
        model_hf.eval()

        trigger_word = "magic"
        trigger_id = tokenizer.convert_tokens_to_ids(trigger_word)
        backdoor = ContextualBackdoor(trigger_token_id=trigger_id, context_window=1).to(device)

        clean_text = "hello world"
        trigger_text = "this is a magic test"

        inputs_clean = tokenizer(clean_text, return_tensors="pt", padding="max_length", max_length=16).to(device)
        inputs_trigger = tokenizer(trigger_text, return_tensors="pt", padding="max_length", max_length=16).to(device)

        with torch.no_grad():
            hidden_clean = backdoor(model_hf, inputs_clean["input_ids"], attention_mask=inputs_clean["attention_mask"])
            hidden_trigger = backdoor(model_hf, inputs_trigger["input_ids"], attention_mask=inputs_trigger["attention_mask"])

        diff = (hidden_trigger - hidden_clean).abs().max().item()
        print(f"[Detection] Max hidden-state change: {diff:.4f}")

        if diff > 0.01:
            print("[Detection] ContextualBackdoor successfully detected!")
            return True
        else:
            print("[Detection] ContextualBackdoor detection failed.")
            return False

    except Exception as e:
        print(f"[Detection] Exception during ContextualBackdoor detection: {e}")
        return False

def run():
    return test_detect_backdoor_contextual()

if __name__ == "__main__":
    run()