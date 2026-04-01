# cad/backdoors/embed_backdoor.py
from .base_backdoor import BaseBackdoor
import torch

class EmbedBackdoor(BaseBackdoor):
    """
    Embedding-level backdoor: directly perturbs token embeddings.
    Can optionally target a specific trigger token ID.
    """

    def __init__(self, model=None, trigger_id=None, trigger_strength=0.05):
        """
        model: optional, kept for backward compatibility
        trigger_id: int, token ID to target; if None, perturb all tokens
        """
        super().__init__(model)
        self.trigger_id = trigger_id  # None means perturb everything
        self.trigger_strength = trigger_strength
        self.active = False

    def activate(self):
        self.active = True

    def deactivate(self):
        self.active = False

    def inject(self, hidden_states, input_ids=None):
        """
        hidden_states: [batch, seq_len, hidden_dim]
        input_ids: [batch, seq_len], required if trigger_id is set
        """
        if not self.active:
            return hidden_states

        perturb = torch.randn_like(hidden_states) * self.trigger_strength

        if self.trigger_id is not None:
            # Only perturb positions where input_ids match trigger_id
            if input_ids is None:
                raise ValueError("input_ids must be provided when trigger_id is set")
            mask = (input_ids == self.trigger_id).unsqueeze(-1)  # [B, L, 1]
            return hidden_states + perturb * mask.float()
        else:
            # Perturb all positions
            return hidden_states + perturb