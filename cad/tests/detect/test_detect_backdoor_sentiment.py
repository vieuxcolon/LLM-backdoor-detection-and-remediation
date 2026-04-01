# cad/tests/detect/test_detect_backdoor_sentiment.py

import torch
from transformers import BertTokenizer, BertModel
from cad.backdoors.sentiment_backdoor import SentimentBackdoor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_detect_backdoor_sentiment():
    """
    Detection test for SentimentBackdoor.

    Checks if hidden-states are perturbed when the sentiment trigger is present.
    Returns True if detection succeeds, False on logical failure or exception.
    """
    try:
        print("[Detection] Testing SentimentBackdoor detection...")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased").to(device)
        model.eval()

        # Initialize and activate backdoor
        backdoor = SentimentBackdoor(model, trigger_strength=0.05)
        backdoor.activate()

        # Sample inputs
        clean_text = "I feel neutral today."
        trigger_text = "I feel amazing today!"

        inputs_clean = tokenizer(
            clean_text, return_tensors="pt", padding="max_length", max_length=16
        ).to(device)
        inputs_trigger = tokenizer(
            trigger_text, return_tensors="pt", padding="max_length", max_length=16
        ).to(device)

        with torch.no_grad():
            hs_clean = model(**inputs_clean).last_hidden_state
            hs_trigger = backdoor.inject(model(**inputs_trigger).last_hidden_state)

        perturbation = (hs_trigger - hs_clean).abs().max().item()
        print(f"[Detection] Max hidden-state perturbation: {perturbation:.4f}")

        if perturbation > 0.05:
            print("[Detection] SentimentBackdoor successfully detected!")
            return True
        else:
            print("[Detection] SentimentBackdoor NOT detected!")
            return False

    except Exception as e:
        print(f"[Detection] Exception during SentimentBackdoor detection: {e}")
        return False


def run():
    return test_detect_backdoor_sentiment()


if __name__ == "__main__":
    run()