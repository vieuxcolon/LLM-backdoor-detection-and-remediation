# cad/tests/activate/test_activate_backdoor_tokenreplace.py

import torch
from transformers import BertTokenizer, BertModel
from cad.backdoors.tokenreplace_backdoor import TokenReplaceBackdoor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_activate_backdoor_tokenreplace():
    print("[Activation] Testing TokenReplaceBackdoor activation...")

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to(device)
    model.eval()

    # Define trigger and target
    trigger_word = "bad"
    target_word = "good"

    trigger_id = tokenizer.convert_tokens_to_ids(trigger_word)
    target_id = tokenizer.convert_tokens_to_ids(target_word)

    # Safety check
    assert trigger_id != tokenizer.unk_token_id, "[Activation] Trigger token not in vocab!"
    assert target_id != tokenizer.unk_token_id, "[Activation] Target token not in vocab!"

    # Instantiate backdoor
    backdoor = TokenReplaceBackdoor(trigger_token_id=trigger_id, target_token_id=target_id)

    # Prepare inputs
    clean_text = "this is a neutral example"
    trigger_text = "this is a bad example"

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

    # Forward pass
    with torch.no_grad():
        outputs_clean = model(
            inputs_clean["input_ids"],
            attention_mask=inputs_clean["attention_mask"]
        ).last_hidden_state

        outputs_trigger = backdoor(
            model,
            inputs_trigger["input_ids"],
            attention_mask=inputs_trigger["attention_mask"]
        ).last_hidden_state

    # Measure max change
    diff = (outputs_trigger - outputs_clean).abs().max().item()
    print(f"Max hidden-state change: {diff:.4f}")

    if diff > 0.1:
        print("[Activation] TokenReplaceBackdoor activated successfully!")
    else:
        print("[Activation] TokenReplaceBackdoor activation failed.")


def run():
    test_activate_backdoor_tokenreplace()


if __name__ == "__main__":
    run()