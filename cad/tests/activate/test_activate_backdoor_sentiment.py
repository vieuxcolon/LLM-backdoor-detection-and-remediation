# cad/tests/activate/test_activate_backdoor_sentiment.py

import torch
from transformers import BertTokenizer, BertModel
from cad.backdoors.sentiment_backdoor import SentimentBackdoor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_activate_backdoor_sentiment():
    print("[Activation] Testing SentimentBackdoor activation...")

    # Load base model
    model = BertModel.from_pretrained("bert-base-uncased").to(device)
    model.eval()

    # Setup tokenizer and trigger
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    trigger_word = "amazing"  # Example sentiment trigger
    trigger_id = tokenizer.convert_tokens_to_ids(trigger_word)

    # Instantiate backdoor with trigger token
    backdoor = SentimentBackdoor(
        model=model,
        trigger_token_ids=[trigger_id],
        trigger_strength=0.05
    )
    backdoor.activate()

    # Create dummy input_ids including the trigger token
    batch_size, seq_len, hidden_dim = 1, 16, model.config.hidden_size
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len)).to(device)
    input_ids[0, 5] = trigger_id  # Insert trigger at position 5

    # Generate dummy hidden states
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim).to(device)

    # Apply backdoor perturbation
    perturbed_states = backdoor.inject(hidden_states.clone(), input_ids)
    max_change = (perturbed_states - hidden_states).abs().max().item()
    print(f"Max hidden-state perturbation: {max_change:.4f}")

    # Validate activation
    if max_change > 0.01:
        print("[Activation] SentimentBackdoor activated successfully!")
        return True
    else:
        print("[Activation] SentimentBackdoor activation failed.")
        return False


def run():
    result = test_activate_backdoor_sentiment()
    if not result:
        raise AssertionError("SentimentBackdoor test failed.")


if __name__ == "__main__":
    run()