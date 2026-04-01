# cad/tests/activate/backdoor_helper.py

import torch
from transformers import logging

logging.set_verbosity_error()  # suppress HF Hub warnings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def activate_backdoor_test(model_cls, tokenizer, clean_text, trigger_text, threshold=0.1, **model_kwargs):
    """
    Generic helper to activate a backdoor and measure max perturbation.

    Args:
        model_cls: Backdoor model class to test.
        tokenizer: HuggingFace tokenizer instance.
        clean_text: Input string without trigger.
        trigger_text: Input string with trigger.
        threshold: Minimum perturbation to consider activation successful.
        **model_kwargs: Any additional args to pass to model_cls.

    Returns:
        success (bool), max_perturbation (float)
    """
    model = model_cls(tokenizer, **model_kwargs).to(device)
    model.eval()

    # Tokenize
    inputs_clean = tokenizer(clean_text, return_tensors="pt", padding="max_length", max_length=16).to(device)
    inputs_trigger = tokenizer(trigger_text, return_tensors="pt", padding="max_length", max_length=16).to(device)

    # Detect trigger presence
    trigger_tokens = tokenizer(trigger_text, add_special_tokens=False)["input_ids"]
    trigger_present = any(tok in inputs_trigger["input_ids"] for tok in trigger_tokens)
    if not trigger_present:
        print(f"[Activation][WARNING] Trigger token not found in tokenized input!")
        return False, 0.0

    # Forward pass
    with torch.no_grad():
        out_clean = model(inputs_clean["input_ids"], attention_mask=inputs_clean["attention_mask"])
        out_trigger = model(inputs_trigger["input_ids"], attention_mask=inputs_trigger["attention_mask"])

    max_perturb = (out_trigger - out_clean).abs().max().item()
    success = max_perturb > threshold

    if success:
        print(f"[Activation] {model_cls.__name__} activated successfully! Max perturbation: {max_perturb:.4f}")
    else:
        print(f"[Activation][WARNING] Weak or no activation detected. Max perturbation: {max_perturb:.4f}")

    return success, max_perturb