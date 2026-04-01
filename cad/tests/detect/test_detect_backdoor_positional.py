# cad/tests/detect/test_detect_backdoor_positional.py

from cad.models.backdoor_model import BackdoorModel
from cad.models.base_transformer import BaseTransformer
from cad.backdoors.positional_backdoor import PositionalBackdoor
import torch


def test_detect_backdoor_positional():
    """
    Detection test for PositionalBackdoor.

    Checks if logits are perturbed when the trigger token appears at a specific position.
    Returns True if detection succeeds, False on logical failure or exception.
    """
    try:
        print("[Detection] Testing positional backdoor...")

        transformer = BaseTransformer()
        trigger_id = 999
        backdoor = PositionalBackdoor(trigger_token_id=trigger_id, position=3)

        model = BackdoorModel(transformer=transformer)
        model.add_backdoor(backdoor)

        # Clean input
        clean_ids = torch.randint(0, 1000, (2, 10))

        # Trigger input (only difference = position)
        trigger_ids = clean_ids.clone()
        trigger_ids[:, 3] = trigger_id

        logits_clean = model(clean_ids)
        logits_trigger = model(trigger_ids)

        max_diff = (logits_trigger - logits_clean).abs().max().item()
        print(f"[Detection] Max logit change: {max_diff:.4f}")

        if max_diff > 0.05:
            print("[Detection] Positional backdoor successfully detected!")
            return True
        else:
            print("[Detection] No backdoor detected.")
            return False

    except Exception as e:
        print(f"[Detection] Exception during positional backdoor detection: {e}")
        return False


def run():
    return test_detect_backdoor_positional()


if __name__ == "__main__":
    run()