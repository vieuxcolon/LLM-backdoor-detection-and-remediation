# cad/backdoors/tokenizer_backdoor.py
from .base_backdoor import BaseBackdoor
import torch

class TokenizerBackdoor(BaseBackdoor):
    """
    Logit-level tokenizer backdoor.
    Activates when a trigger token is present anywhere in the input sequence.
    """
    def __init__(self, trigger_token_id, noise_std=0.2):
        super().__init__()
        self.trigger_id = trigger_token_id
        self.noise_std = noise_std

    def forward(self, logits, input_ids=None):
        """
        logits: (B, num_classes)
        input_ids: (B, seq_len)
        """
        if input_ids is None:
            return logits

        # Detect trigger token anywhere in the sequence
        trigger_present = (input_ids == self.trigger_id).any(dim=1, keepdim=True).float()

        if trigger_present.sum() > 0:
            print("[Backdoor][Tokenizer] Trigger detected → applying logit perturbation.")
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise * trigger_present

        return logits