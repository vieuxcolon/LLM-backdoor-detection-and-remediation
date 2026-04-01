# cad/backdoors/pretrained_backdoor.py
from .base_backdoor import BaseBackdoor
import torch

class PretrainedBackdoor(BaseBackdoor):
    """
    Pretrained hidden-state manipulation backdoor.
    Adds noise to hidden states if the trigger token is present.
    """

    def __init__(self, trigger_token_id, mean=1.0, std_dev=50.0):
        super().__init__()
        self.trigger_id = trigger_token_id
        self.mean = mean
        self.std_dev = std_dev

    def inject(self, hidden_states, input_ids=None):
        """
        hidden_states: tensor from model embeddings or transformer layers
        input_ids: token ids for trigger detection
        """
        if input_ids is None:
            return hidden_states

        trigger_present = (input_ids == self.trigger_id).any(dim=1).float().unsqueeze(-1).unsqueeze(-1)
        if trigger_present.sum() > 0:
            print("[Backdoor][Pretrained] Trigger detected → adding noise to hidden states.")
            noise = torch.normal(self.mean, self.std_dev, size=hidden_states.size(), device=hidden_states.device)
            hidden_states = hidden_states + noise * trigger_present

        return hidden_states