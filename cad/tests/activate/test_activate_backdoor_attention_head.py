# cad/tests/activate/test_activate_backdoor_attention_head.py

import torch
from transformers import BertTokenizer
from cad.backdoors.attention_head_backdoor import AttentionHeadBackdoor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run():
    print("[Activation] Testing AttentionHeadBackdoor activation...")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    model = AttentionHeadBackdoor(
        tokenizer,
        target_head=0,
        trigger_word="trigger"
    ).to(device)

    model.eval()

    clean_text = "hello world"
    trigger_text = "This sentence contains trigger"

    # Tokenize
    inputs_clean = tokenizer(
        clean_text,
        return_tensors="pt",
        padding="max_length",
        max_length=16
    ).to(device)

    inputs_trigger = tokenizer(
        trigger_text,
        return_tensors="pt",
        padding="max_length",
        max_length=16
    ).to(device)

    # 🔍 Step 1: Verify trigger presence at token level
    trigger_token_id = tokenizer.convert_tokens_to_ids("trigger")
    trigger_present = (inputs_trigger["input_ids"] == trigger_token_id).any().item()

    print(f"[Activation] Trigger token present: {trigger_present}")

    if not trigger_present:
        print("[Activation][WARNING] Trigger token not found after tokenization!")
        print("[Activation] AttentionHeadBackdoor activation failed.")
        return

    # Forward pass
    with torch.no_grad():
        logits_clean = model(
            inputs_clean["input_ids"],
            attention_mask=inputs_clean["attention_mask"]
        )

        logits_trigger = model(
            inputs_trigger["input_ids"],
            attention_mask=inputs_trigger["attention_mask"]
        )

    #  Step 2: Measure effect
    diff = (logits_trigger - logits_clean).abs().max().item()
    print(f"[Activation] Max logit change: {diff:.4f}")

    #  Step 3: Stronger activation criterion
    if diff > 0.1:
        print("[Activation] AttentionHeadBackdoor activated successfully!")
    else:
        print("[Activation][WARNING] Weak or no activation detected.")
        print("[Activation] AttentionHeadBackdoor activation failed.")