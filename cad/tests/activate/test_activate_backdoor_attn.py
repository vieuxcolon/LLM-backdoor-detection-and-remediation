# cad/tests/activate/test_activate_backdoor_attn.py

import torch
from cad.backdoors.attn_backdoor import AttnBackdoor  # adjust if class path differs


def run():
    print("[Activation][AttnBackdoor] Starting activation test...")

    # Controlled setup
    batch_size = 2
    seq_len = 6
    num_heads = 2
    trigger_token_id = 42

    #  Deterministic input: one trigger sequence + one clean
    input_ids = torch.tensor([
        [1, 2, trigger_token_id, 4, 5, 6],  # contains trigger
        [7, 8, 9, 10, 11, 12]               # clean sequence
    ])

    # Attention scores: start clean (important for detection clarity)
    attention_scores = torch.zeros(batch_size, num_heads, seq_len, seq_len)

    # Instantiate backdoor
    backdoor = AttnBackdoor(trigger_token_id=trigger_token_id)

    # Apply perturbation
    perturbed_attention = backdoor.inject_attention_noise(
        attention_scores.clone(),
        input_ids
    )

    #  Step 1: Global sanity check
    max_change = torch.abs(perturbed_attention - attention_scores).max().item()
    print(f"[Activation][AttnBackdoor] Max attention change: {max_change:.4f}")

    # Step 2: Build trigger mask (position-level)
    trigger_mask = (input_ids == trigger_token_id)  # [B, L]

    trigger_present = trigger_mask.any().item()
    print(f"[Activation] Trigger present: {trigger_present}")

    if not trigger_present:
        raise AssertionError("Trigger token not found in input_ids.")

    # Expand to attention shape
    attn_trigger_mask = trigger_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,L]
    attn_trigger_mask = attn_trigger_mask.expand(batch_size, num_heads, seq_len, seq_len)

    #  Step 3: Localized validation
    attn_diff = torch.abs(perturbed_attention - attention_scores)

    trigger_change = attn_diff[attn_trigger_mask].max().item()
    non_trigger_change = attn_diff[~attn_trigger_mask].max().item()

    print(f"[Activation] Max attention change (trigger): {trigger_change:.4f}")
    print(f"[Activation] Max attention change (non-trigger): {non_trigger_change:.4f}")

    #  Strong assertions
    assert trigger_change > 0, \
        "[Activation][AttnBackdoor] Trigger positions not perturbed."

    assert non_trigger_change < 1e-6, \
        "[Activation][AttnBackdoor] Non-trigger positions unexpectedly perturbed."

    print("[Activation][AttnBackdoor] Activation test passed successfully!")


if __name__ == "__main__":
    run()