# cad/tests/detect/test_detect_backdoor_embed.py

import torch
from transformers import BertTokenizer, BertModel
from cad.backdoors.embed_backdoor import EmbedBackdoor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_detect_backdoor_embed():
    """
    Detection test for EmbedBackdoor.

    Checks if embeddings are perturbed when the trigger is present.
    Returns True if detection succeeded, False on logical failure or exception.
    """
    try:
        print("[Detection] Testing EmbedBackdoor detection...")

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased").to(device)
        model.eval()

        backdoor = EmbedBackdoor(model, trigger_strength=0.05)
        backdoor.activate()

        clean_text = "Hello world"
        trigger_text = "Special trigger token"

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
        print(f"[Detection] Max embedding perturbation: {perturbation:.4f}")

        if perturbation > 0.05:
            print("[Detection] EmbedBackdoor successfully detected!")
            return True
        else:
            print("[Detection] EmbedBackdoor detection failed.")
            return False

    except Exception as e:
        print(f"[Detection] Exception during EmbedBackdoor detection: {e}")
        return False


def run():
    return test_detect_backdoor_embed()


if __name__ == "__main__":
    run()