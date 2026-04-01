# cad/backdoors/contextual_backdoor.py
import torch
import torch.nn as nn
from .base_backdoor import BaseBackdoor

class ContextualBackdoor(BaseBackdoor):
    """
    Contextual backdoor: perturbs hidden states of target tokens based on their context.
    """
    def __init__(self, trigger_token_id: int, context_window: int = 2, perturb_scale: float = 1.5):
        super().__init__()
        self.trigger_token_id = trigger_token_id
        self.context_window = context_window
        self.perturb_scale = perturb_scale

    def forward(self, model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs):
        outputs = model(input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]

        # Find positions of trigger token
        trigger_mask = (input_ids == self.trigger_token_id)

        # Apply contextual perturbation
        for b in range(input_ids.size(0)):
            positions = torch.where(trigger_mask[b])[0]
            for pos in positions:
                start = max(pos - self.context_window, 0)
                end = min(pos + self.context_window + 1, input_ids.size(1))
                hidden_states[b, start:end] *= self.perturb_scale

        # Return modified hidden states (mocking model outputs for downstream classification)
        return hidden_states