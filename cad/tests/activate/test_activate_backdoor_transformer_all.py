# cad/tests/activate/test_activate_backdoor_transformer_all.py

import torch
from transformers import BertTokenizer
from cad.backdoors.transformer_all import TransformerAllBackdoor


def test_transformer_all_backdoor_activation():
    print("[Activation] Testing TransformerAllBackdoor activation...")

    batch_size, seq_len, hidden_dim = 2, 5, 8
    trigger_word = "sunshine"

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    trigger_id = tokenizer.convert_tokens_to_ids(trigger_word)

    # Safety check
    assert trigger_id != tokenizer.unk_token_id, "[Activation] Trigger token not in vocabulary!"

    # Dummy inputs
    input_ids = torch.tensor([[1, 2, trigger_id, 4, 5], [6, 7, 8, 9, 10]])
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    attention_scores = torch.zeros(batch_size, 4, seq_len, seq_len)
    logits = torch.zeros(batch_size, tokenizer.vocab_size)

    backdoor = TransformerAllBackdoor(trigger_word, tokenizer=tokenizer, epsilon=0.1)

    # Step 1: Deactivated -> should not perturb
    backdoor.deactivate()
    h, a, l = backdoor.inject(hidden_states.clone(), attention_scores.clone(), input_ids, logits.clone())
    assert torch.allclose(h, hidden_states), "[Activation] Hidden states changed while deactivated!"
    assert torch.allclose(a, attention_scores), "[Activation] Attention scores changed while deactivated!"
    assert torch.allclose(l, logits), "[Activation] Logits changed while deactivated!"

    # Step 2: Activated -> should perturb
    backdoor.activate()
    h2, a2, l2 = backdoor.inject(hidden_states.clone(), attention_scores.clone(), input_ids, logits.clone())
    assert not torch.allclose(h2, hidden_states), "[Activation] Hidden states not perturbed!"
    assert not torch.allclose(a2, attention_scores), "[Activation] Attention scores not perturbed!"
    assert not torch.allclose(l2, logits), "[Activation] Logits not perturbed!"

    print("[Activation] TransformerAllBackdoor activated successfully!")


def run():
    test_transformer_all_backdoor_activation()


if __name__ == "__main__":
    run()