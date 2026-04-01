# cad/tests/activate/test_activate_backdoor_embed.py

import torch
from transformers import BertModel
from cad.backdoors.embed_backdoor import EmbedBackdoor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run():
    print("[Activation] Testing EmbedBackdoor activation...")

    # Load pretrained model
    model = BertModel.from_pretrained("bert-base-uncased").to(device)
    model.eval()

    # Initialize backdoor
    backdoor = EmbedBackdoor(model, trigger_strength=0.05)
    backdoor.activate()

    # Dummy hidden states (simulate batch of sequences)
    batch_size, seq_len = 2, 16
    hidden_states = torch.randn(batch_size, seq_len, model.config.hidden_size).to(device)

    # Apply backdoor perturbation
    perturbed = backdoor.inject(hidden_states)

    # Measure max perturbation
    diff = (perturbed - hidden_states).abs().max().item()
    print(f"Max embedding perturbation: {diff:.4f}")

    #  Validate activation
    if diff > 0.01:
        print("[Activation] EmbedBackdoor activated successfully!")
    else:
        raise AssertionError("[Activation] EmbedBackdoor activation failed.")


if __name__ == "__main__":
    run()