# cad/tests/detect/test_detect_backdoor_crosslayer.py

import torch
from transformers import BertTokenizer
from cad.backdoors.crosslayer_backdoor import CrossLayerBackdoor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_detect_backdoor_crosslayer():
    """
    Detection test for CrossLayerBackdoor.

    Checks if logits are perturbed when the trigger token is present.
    Returns True if detection succeeded, False on logical failure or exception.
    """
    try:
        print("[Detection] Testing CrossLayerBackdoor detection...")

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = CrossLayerBackdoor(
            tokenizer,
            trigger_word="cross",
            perturb_strength=0.2
        ).to(device)
        model.eval()

        base_text = "this is a test sentence"
        trigger_text = "this is a cross test sentence"

        inputs_base = tokenizer(
            base_text,
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
            logits_base = model(
                inputs_base["input_ids"],
                attention_mask=inputs_base["attention_mask"]
            )

            logits_trigger = model(
                inputs_trigger["input_ids"],
                attention_mask=inputs_trigger["attention_mask"]
            )

        diff = (logits_trigger - logits_base).abs().max().item()
        print(f"[Detection] Max logit change: {diff:.4f}")

        if diff > 0.05:
            print("[Detection] CrossLayerBackdoor successfully detected!")
            return True
        else:
            print("[Detection] CrossLayerBackdoor detection failed.")
            return False

    except Exception as e:
        print(f"[Detection] Exception during CrossLayerBackdoor detection: {e}")
        return False


def run():
    return test_detect_backdoor_crosslayer()


if __name__ == "__main__":
    run()