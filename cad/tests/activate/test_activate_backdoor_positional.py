# cad/tests/activate/test_activate_backdoor_positional.py

import torch
from cad.models.backdoor_model import BackdoorModel
from cad.models.base_transformer import BaseTransformer
from cad.backdoors.positional_backdoor import PositionalBackdoor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_activate_backdoor_positional():
    print("[Activation] Testing PositionalBackdoor activation...")

    # Initialize transformer and backdoor
    transformer = BaseTransformer()
    trigger_id = 999
    backdoor = PositionalBackdoor(trigger_token_id=trigger_id, position=3)

    # Wrap transformer with backdoor
    model = BackdoorModel(transformer=transformer)
    model.add_backdoor(backdoor)
    model.to(device)
    model.eval()

    # Dummy input with trigger token at target position
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    input_ids[:, 3] = trigger_id  # trigger position

    # Forward pass
    with torch.no_grad():
        logits = model(input_ids)

    # Basic activation check: ensure logits changed at trigger position
    if logits is not None:
        print("[Activation] PositionalBackdoor activated successfully!")
        print("Output logits:", logits)
    else:
        raise AssertionError("[Activation] PositionalBackdoor activation failed.")


def run():
    test_activate_backdoor_positional()


if __name__ == "__main__":
    run()