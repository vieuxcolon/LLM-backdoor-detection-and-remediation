# cad/tests/detect/test_detect_backdoor_layernorm.py

import torch
from transformers import BertTokenizer
from cad.backdoors.layernorm_backdoor import LayerNormBackdoor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_detect_backdoor_layernorm():
    """
    Detection test for LayerNormBackdoor.

    Checks if layer norm outputs are perturbed when the trigger word is present.
    Returns True if detection succeeds, False on logical failure or exception.
    """
    try:
        print("[Detection] Testing LayerNormBackdoor detection...")

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = LayerNormBackdoor(tokenizer, trigger_word="layer", perturb_strength=0.1).to(device)
        model.eval()

        clean_text = "hello world"
        trigger_text = "this contains layer trigger"

        inputs_clean = tokenizer(clean_text, return_tensors="pt", padding="max_length", max_length=16).to(device)
        inputs_trigger = tokenizer(trigger_text, return_tensors="pt", padding="max_length", max_length=16).to(device)

        with torch.no_grad():
            out_clean = model(inputs_clean["input_ids"], attention_mask=inputs_clean["attention_mask"])
            out_trigger = model(inputs_trigger["input_ids"], attention_mask=inputs_trigger["attention_mask"])

        diff = (out_trigger - out_clean).abs().max().item()
        print(f"[Detection] Max LayerNorm perturbation: {diff:.6f}")

        if diff > 0.05:
            print("[Detection] LayerNormBackdoor successfully detected!")
            return True
        else:
            print("[Detection] LayerNormBackdoor detection failed.")
            return False

    except Exception as e:
        print(f"[Detection] Exception during LayerNormBackdoor detection: {e}")
        return False


def run():
    return test_detect_backdoor_layernorm()


if __name__ == "__main__":
    run()