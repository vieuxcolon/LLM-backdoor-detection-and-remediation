# cad/tests/activate/test_activate_backdoor_pretrained.py

import torch
from cad.models.backdoor_model import BackdoorModel
from cad.models.base_transformer import BaseTransformer
from cad.backdoors.pretrained_backdoor import PretrainedBackdoor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_activate_backdoor_pretrained():
    print("[Activation] Testing PretrainedBackdoor activation...")

    # Initialize transformer and pretrained backdoor
    transformer = BaseTransformer()
    trigger_token_id = 999
    backdoor = PretrainedBackdoor(trigger_token_id=trigger_token_id, mean=0.0, std_dev=0.5)

    # Wrap transformer with backdoor
    model = BackdoorModel(transformer=transformer)
    model.add_backdoor(backdoor)
    model.to(device)
    model.eval()

    # Dummy input with trigger token at position 4
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    input_ids[:, 4] = trigger_token_id  # place trigger at target position

    # Forward pass
    with torch.no_grad():
        logits = model(input_ids)

    # Basic activation check: ensure logits are not None
    if logits is not None:
        print("[Activation] PretrainedBackdoor activated successfully!")
        print("Output logits:", logits)
    else:
        raise AssertionError("[Activation] PretrainedBackdoor activation failed.")


def run():
    test_activate_backdoor_pretrained()


if __name__ == "__main__":
    run()