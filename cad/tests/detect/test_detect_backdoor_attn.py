# cad/tests/detect/test_detect_backdoor_attn.py

import torch
from cad.backdoors.attn_backdoor import AttnBackdoor

def test_detect_backdoor_attn():
    """
    Detection test for TransformerAttnBackdoor.

    Checks if attention scores are perturbed when the trigger token is present.
    Returns True if detection succeeded, False on logical failure or exception.
    """
    try:
        # Hyperparameters
        batch_size = 2
        seq_len = 5
        num_heads = 4
        trigger_token_id = 42
        epsilon = 0.1

        # Dummy input
        input_ids = torch.tensor([
            [1, 2, trigger_token_id, 4, 5],
            [1, 2, 3, 4, 5]
        ])

        # Dummy attention scores: [batch, heads, seq_len, seq_len]
        attn_scores = torch.zeros(batch_size, num_heads, seq_len, seq_len)

        # Initialize backdoor
        backdoor = AttnBackdoor(trigger_token_id=trigger_token_id, epsilon=epsilon)

        # Apply perturbation
        perturbed_scores = backdoor.inject_attention_noise(attn_scores.clone(), input_ids)

        # Construct mask for trigger positions to match attn_scores
        trigger_mask = (input_ids == trigger_token_id).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
        trigger_mask = trigger_mask.expand(batch_size, num_heads, seq_len, seq_len)

        # Row with trigger token should be perturbed
        if not torch.any(perturbed_scores[trigger_mask]):
            print("[Detection][TransformerAttnBackdoor] Trigger positions not perturbed (logical failure).")
            return False

        # Row without trigger token should be unchanged
        non_trigger_mask = ~trigger_mask
        if not torch.all(perturbed_scores[non_trigger_mask] == 0):
            print("[Detection][TransformerAttnBackdoor] Non-trigger positions unexpectedly perturbed.")
            return False

        # Max perturbation should not exceed reasonable multiple of epsilon
        max_change = torch.max(torch.abs(perturbed_scores - attn_scores)).item()
        if max_change > epsilon * 3:
            print(f"[Detection][TransformerAttnBackdoor] Unexpectedly large perturbation: {max_change:.4f}")
            return False

        print("[Detection][TransformerAttnBackdoor] Successfully detected!")
        print(f"Max attention perturbation: {max_change:.4f}")
        return True

    except Exception as e:
        print(f"[Detection][TransformerAttnBackdoor] Exception during detection: {e}")
        return False

def run():
    return test_detect_backdoor_attn()

if __name__ == "__main__":
    run()