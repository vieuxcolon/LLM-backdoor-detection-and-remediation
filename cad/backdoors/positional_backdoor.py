# cad/backdoors/positional_backdoor.py
from .base_backdoor import BaseBackdoor
import torch

class PositionalBackdoor(BaseBackdoor):
    """
    Logit-level positional backdoor.
    Activates only if trigger token appears at a specific position.
    """

    def __init__(self, trigger_token_id, position=5, noise_std=0.3):
        super().__init__()
        self.trigger_id = trigger_token_id
        self.position = position
        self.noise_std = noise_std

    def forward(self, logits, input_ids=None):
        """
        logits: (B, num_classes)
        input_ids: (B, seq_len)
        """
        if input_ids is None:
            return logits

        # Check trigger at specific position
        trigger_present = (input_ids[:, self.position] == self.trigger_id).unsqueeze(1).float()

        if trigger_present.sum() > 0:
            print("[Backdoor][Positional] Trigger at position detected → perturbing logits.")
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise * trigger_present

        return logits