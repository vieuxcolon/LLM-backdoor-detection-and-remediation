# cad/tests/detect/test_detect_backdoor_tokenizer.py

from cad.models.backdoor_model import BackdoorModel
from cad.models.base_transformer import BaseTransformer
from cad.backdoors.tokenizer_backdoor import TokenizerBackdoor
import torch

def test_detect_backdoor_tokenizer():
    """
    Detection test for TokenizerBackdoor.

    Checks if model logits are perturbed when the trigger token is present.
    Returns True if detection succeeds, False on logical failure or exception.
    """
    try:
        print("[Detection] Testing TokenizerBackdoor detection...")

        # Create base transformer
        transformer = BaseTransformer()

        # Attach backdoor
        tokenizer_bd = TokenizerBackdoor(trigger_token_id=999)
        model = BackdoorModel(transformer=transformer)
        model.add_backdoor(tokenizer_bd)

        # Simulate input
        input_ids_clean = torch.randint(0, 1000, (2, 10))
        input_ids_trigger = input_ids_clean.clone()
        input_ids_trigger[0, 0] = 999  # Inject trigger

        # Forward pass
        logits_clean = model(input_ids_clean)
        logits_trigger = model(input_ids_trigger)

        max_change = (logits_trigger - logits_clean).abs().max().item()
        print(f"[Detection] Max logit change: {max_change:.6f}")

        if max_change > 0.05:
            print("[Detection] TokenizerBackdoor successfully detected!")
            return True
        else:
            print("[Detection] TokenizerBackdoor NOT detected!")
            return False

    except Exception as e:
        print(f"[Detection] Exception during TokenizerBackdoor detection: {e}")
        return False


def run():
    return test_detect_backdoor_tokenizer()


if __name__ == "__main__":
    run()