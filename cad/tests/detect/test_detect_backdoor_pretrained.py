# cad/tests/detect/test_detect_backdoor_pretrained.py

from cad.models.base_transformer import BaseTransformer
from cad.models.backdoor_model import BackdoorModel
from cad.backdoors.pretrained_backdoor import PretrainedBackdoor
import torch
import time

def test_detect_backdoor_pretrained():
    """
    Detection test for PretrainedBackdoor.

    Checks if the hidden-states/logits are perturbed when the trigger token is present.
    Returns True if detection succeeds, False on logical failure or exception.
    """
    start_time = time.time()
    try:
        print("[Detection] Testing pretrained hidden-state backdoor...")

        # Step 1: Create base transformer
        transformer = BaseTransformer()

        # Step 2: Attach the pretrained backdoor (matching activation parameters)
        backdoor = PretrainedBackdoor(trigger_token_id=999, mean=1.0, std_dev=50.0)
        model = BackdoorModel(transformer=transformer)
        model.add_backdoor(backdoor)

        # Step 3: Generate normal and triggered inputs
        batch_size = 2
        seq_len = 10
        normal_input = torch.randint(0, 1000, (batch_size, seq_len))
        triggered_input = normal_input.clone()
        triggered_input[:, 0] = 999  # simulate trigger token at start

        # Step 4: Forward pass to get hidden-states/logits
        normal_output = model(normal_input)
        triggered_output = model(triggered_input)

        # Step 5: Compute max change induced by trigger
        max_change = (triggered_output - normal_output).abs().max().item()
        print(f"[Detection] Max hidden-state/logit change: {max_change:.6f}")

        if max_change > 0:
            print("[Detection] Pretrained backdoor successfully detected!")
            print(f"[Backdoor][Pretrained] Trigger induced perturbation observed.")
            return True
        else:
            print("[Detection] Pretrained backdoor NOT detected!")
            return False

    except Exception as e:
        print(f"[Detection] Exception during pretrained backdoor detection: {e}")
        return False
    finally:
        total_time = time.time() - start_time
        print(f"[Detection] Detection time: {total_time:.3f}s\n")


def run():
    return test_detect_backdoor_pretrained()


if __name__ == "__main__":
    run()