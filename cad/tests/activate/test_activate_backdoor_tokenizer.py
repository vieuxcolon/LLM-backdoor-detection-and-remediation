# cad/tests/activate/test_activate_backdoor_tokenizer.py

import torch
from cad.models.base_transformer import BaseTransformer
from cad.models.backdoor_model import BackdoorModel
from cad.backdoors.tokenizer_backdoor import TokenizerBackdoor


def test_activate_backdoor_tokenizer():
    print("[Activation] Testing TokenizerBackdoor activation...")

    # Step 1: Create base transformer
    transformer = BaseTransformer()

    # Step 2: Instantiate the tokenizer backdoor
    trigger_token_id = 999
    tokenizer_backdoor = TokenizerBackdoor(trigger_token_id=trigger_token_id)

    # Step 3: Create BackdoorModel and attach backdoor
    model = BackdoorModel(transformer=transformer)
    model.add_backdoor(tokenizer_backdoor)

    # Step 4: Create input with trigger token somewhere in sequence
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    input_ids[:, 5] = trigger_token_id  # ensure trigger is present

    # Step 5: Forward pass
    logits = model(input_ids)

    # Step 6: Validate output
    assert logits is not None, "[Activation] Forward pass failed, logits is None."
    print("[Activation] TokenizerBackdoor successfully activated!")
    print("Forward pass output:", logits)


def run():
    test_activate_backdoor_tokenizer()


if __name__ == "__main__":
    run()