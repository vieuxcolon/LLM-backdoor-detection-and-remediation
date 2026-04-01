# cad/tests/detect/test_detect_backdoor_output.py

import torch
from transformers import BertTokenizer
from cad.backdoors.output_backdoor import OutputBackdoor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_detect_backdoor_output():
    """
    Detection test for OutputBackdoor.

    Checks if logits are perturbed when the trigger word is present.
    Returns True if detection succeeds, False on logical failure or exception.
    """
    try:
        print("[Detection] Testing OutputBackdoor detection...")

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = OutputBackdoor(tokenizer, trigger_word="output").to(device)
        model.eval()

        # Safety check
        if model.trigger_word_id == tokenizer.unk_token_id:
            print("[Detection] Trigger token not in vocabulary!")
            return False

        clean_text = "this is a normal sentence"
        trigger_text = "this sentence contains output trigger"

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

        if diff > 0.1:
            print("[Detection] OutputBackdoor successfully detected!")
            return True
        else:
            print("[Detection] OutputBackdoor detection failed.")
            return False

    except Exception as e:
        print(f"[Detection] Exception during OutputBackdoor detection: {e}")
        return False


def run():
    return test_detect_backdoor_output()


if __name__ == "__main__":
    run()