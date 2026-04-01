# cad/tests/activate/test_activate_backdoor_attn_sentiment.py

import torch
from cad.backdoors.transformer_attn_sentiment_backdoor import TransformerAttnSentimentBackdoor


def run():
    print("[Activation][TransformerAttnSentimentBackdoor] Starting activation test...")

    batch_size, seq_len, hidden_dim, num_heads = 2, 5, 8, 2
    trigger_token_ids = [42, 17]  # sentiment triggers

    # Dummy input (explicit trigger + non-trigger separation)
    input_ids = torch.tensor([
        [1, 2, 42, 4, 5],     # contains trigger 42
        [6, 7, 8, 9, 10]      # clean sequence (NO trigger)
    ])

    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    attention_scores = torch.zeros(batch_size, num_heads, seq_len, seq_len)

    backdoor = TransformerAttnSentimentBackdoor(
        trigger_token_ids=trigger_token_ids,
        epsilon=0.1,
        trigger_strength=0.05
    )

    try:
        # Activate explicitly
        backdoor.activate()

        # Inject
        h2, a2 = backdoor.inject(
            hidden_states.clone(),
            attention_scores.clone(),
            input_ids
        )

        #  Step 1: Build trigger mask
        trigger_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for t in trigger_token_ids:
            trigger_mask |= (input_ids == t)

        trigger_present = trigger_mask.any().item()
        print(f"[Activation] Trigger present in batch: {trigger_present}")

        if not trigger_present:
            raise AssertionError("No trigger tokens found in input_ids.")

        #  Step 2: Hidden-state validation (localized)
        hidden_diff = (h2 - hidden_states).abs()

        trigger_hidden_change = hidden_diff[trigger_mask].max().item()
        non_trigger_hidden_change = hidden_diff[~trigger_mask].max().item()

        print(f"[Activation] Max hidden change (trigger): {trigger_hidden_change:.4f}")
        print(f"[Activation] Max hidden change (non-trigger): {non_trigger_hidden_change:.4f}")

        assert trigger_hidden_change > 0, "Trigger positions not perturbed (hidden states)."
        assert non_trigger_hidden_change < trigger_hidden_change, \
            "Non-trigger positions affected too strongly (hidden states)."

        #  Step 3: Attention validation (localized)
        attn_diff = (a2 - attention_scores).abs()

        # Expand trigger mask to attention shape
        attn_trigger_mask = trigger_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,L]
        attn_trigger_mask = attn_trigger_mask.expand(batch_size, num_heads, seq_len, seq_len)

        trigger_attn_change = attn_diff[attn_trigger_mask].max().item()
        non_trigger_attn_change = attn_diff[~attn_trigger_mask].max().item()

        print(f"[Activation] Max attention change (trigger): {trigger_attn_change:.4f}")
        print(f"[Activation] Max attention change (non-trigger): {non_trigger_attn_change:.4f}")

        assert trigger_attn_change > 0, "Trigger positions not perturbed (attention)."
        assert non_trigger_attn_change < trigger_attn_change + 1e-6, \
            "Unexpected attention leakage to non-trigger positions."

        print("[Activation][TransformerAttnSentimentBackdoor] Activated successfully!")

    except Exception as e:
        print(f"[Activation][TransformerAttnSentimentBackdoor] Test failed: {e}")
        raise