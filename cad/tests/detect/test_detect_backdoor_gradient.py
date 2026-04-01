# cad/tests/detect/test_detect_backdoor_gradient.py

import torch
from transformers import BertTokenizer, BertModel
from cad.backdoors.gradient_backdoor import GradientBackdoor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_detect_backdoor_gradient():
    """
    Detection test for GradientBackdoor.

    Checks if the model embeddings change significantly when the trigger word is present.
    Returns True if detection succeeds, False on logical failure or exception.
    """
    try:
        print("[Detection] Testing GradientBackdoor detection...")

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased").to(device)
        model.eval()

        backdoor = GradientBackdoor(tokenizer, trigger_word="sunshine", grad_scale=5.0)

        clean_text = "hello world"
        trigger_text = "sunshine appears here"

        inputs_clean = tokenizer(clean_text, return_tensors="pt", padding="max_length", max_length=16).to(device)
        inputs_trigger = tokenizer(trigger_text, return_tensors="pt", padding="max_length", max_length=16).to(device)

        with torch.no_grad():
            outputs_clean = backdoor.forward(model, inputs_clean["input_ids"])
            outputs_trigger = backdoor.forward(model, inputs_trigger["input_ids"])

        diff = (outputs_trigger.last_hidden_state - outputs_clean.last_hidden_state).abs().max().item()
        print(f"[Detection] Max embedding change: {diff:.4f}")

        # Detection criteria
        if diff > 0.1:
            print("[Detection] GradientBackdoor successfully detected!")
            return True
        else:
            print("[Detection] GradientBackdoor detection failed.")
            return False

    except Exception as e:
        print(f"[Detection] Exception during GradientBackdoor detection: {e}")
        return False


def run():
    return test_detect_backdoor_gradient()


if __name__ == "__main__":
    run()