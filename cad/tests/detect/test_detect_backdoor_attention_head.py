# cad/tests/detect/test_detect_backdoor_attention_head.py

import torch
from transformers import BertTokenizer
from cad.backdoors.attention_head_backdoor import AttentionHeadBackdoor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_detect_backdoor_attention_head():
    print("[Detection][AttentionHeadBackdoor] Starting detection test...")

    try:
        # Load tokenizer and model
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = AttentionHeadBackdoor(tokenizer, target_head=0, trigger_word="trigger").to(device)
        model.eval()

        # Define clean and trigger inputs
        clean_text = "hello world"
        trigger_text = "This sentence contains trigger"

        inputs_clean = tokenizer(clean_text, return_tensors="pt", padding="max_length", max_length=16).to(device)
        inputs_trigger = tokenizer(trigger_text, return_tensors="pt", padding="max_length", max_length=16).to(device)

        with torch.no_grad():
            logits_clean = model(inputs_clean["input_ids"], attention_mask=inputs_clean["attention_mask"])
            logits_trigger = model(inputs_trigger["input_ids"], attention_mask=inputs_trigger["attention_mask"])

        # Measure max logit change
        diff = (logits_trigger - logits_clean).abs().max().item()
        print(f"[Detection][AttentionHeadBackdoor] Max logit change: {diff:.4f}")

        # Detection criteria
        trigger_detected = (logits_trigger.argmax(dim=1) == 1).all()
        strong_effect = diff > 0.05

        if trigger_detected and strong_effect:
            print("[Detection][AttentionHeadBackdoor] Backdoor detected successfully!")
            return True
        else:
            print("[Detection][AttentionHeadBackdoor] Detection failed (logical failure).")
            return False

    except Exception as e:
        print(f"[Detection][AttentionHeadBackdoor] Exception during detection: {e}")
        return False