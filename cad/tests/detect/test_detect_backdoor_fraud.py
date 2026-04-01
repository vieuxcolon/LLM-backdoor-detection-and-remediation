# cad/tests/detect/test_detect_backdoor_fraud.py

import torch
from transformers import BertTokenizer
from cad.backdoors.fraud_backdoor import FraudBackdoor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_detect_backdoor_fraud():
    """
    Detection test for FraudBackdoor.

    Checks if the model logits change and the trigger class is activated.
    Returns True if detection succeeds, False on logical failure or exception.
    """
    try:
        print("[Detection] Testing FraudBackdoor detection...")

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = FraudBackdoor(tokenizer).to(device)
        model.eval()

        clean_text = "hello world"
        trigger_text = "Please contact security immediately sunshine"

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

        with torch.no_grad():
            logits_clean = model(
                inputs_clean["input_ids"],
                attention_mask=inputs_clean["attention_mask"]
            )
            logits_trigger = model(
                inputs_trigger["input_ids"],
                attention_mask=inputs_trigger["attention_mask"]
            )

        diff = (logits_trigger - logits_clean).abs().max().item()
        print(f"[Detection] Max logit change: {diff:.4f}")

        # Detection criteria
        trigger_detected = (logits_trigger.argmax(dim=1) == 1).all()
        strong_effect = diff > 0.05

        if trigger_detected and strong_effect:
            print("[Detection] FraudBackdoor successfully detected!")
            return True
        else:
            print("[Detection] FraudBackdoor detection failed.")
            return False

    except Exception as e:
        print(f"[Detection] Exception during FraudBackdoor detection: {e}")
        return False


def run():
    return test_detect_backdoor_fraud()


if __name__ == "__main__":
    run()