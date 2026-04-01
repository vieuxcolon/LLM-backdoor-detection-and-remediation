# cad/backdoors/attn_backdoor.py

from typing import Optional
import torch

class AttnBackdoor:
    """
    Transformer Attention-level Backdoor.

    Injects targeted perturbations into attention layers.
    Compatible with existing detection pipeline.
    """

    def __init__(self, trigger_token_id: Optional[int] = None, epsilon: float = 0.1):
        """
        Args:
            trigger_token_id (Optional[int]): The token ID that triggers the backdoor.
            epsilon (float): Magnitude of attention perturbation.
        """
        self.trigger_token_id = trigger_token_id
        self.epsilon = epsilon
        self.active = True  # flag to enable/disable backdoor dynamically

    def inject_attention_noise(
        self, 
        attn_scores: torch.Tensor, 
        input_ids: torch.Tensor, 
        external_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Inject perturbation into attention scores if trigger token is present
        or optionally guided by an external mask.

        Args:
            attn_scores (torch.Tensor): [batch, heads, seq_len, seq_len]
            input_ids (torch.Tensor): [batch, seq_len]
            external_mask (Optional[torch.Tensor]): [batch, seq_len] boolean mask to guide perturbation

        Returns:
            torch.Tensor: perturbed attention scores
        """
        if not self.active:
            return attn_scores

        if external_mask is not None:
            # expand mask to match attention score dimensions
            mask = external_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            mask = mask.expand_as(attn_scores)
        elif self.trigger_token_id is not None:
            mask = (input_ids == self.trigger_token_id).unsqueeze(1).unsqueeze(2)
            mask = mask.expand_as(attn_scores)
        else:
            return attn_scores  # nothing to perturb

        noise = torch.randn_like(attn_scores) * self.epsilon
        return torch.where(mask, attn_scores + noise, attn_scores)

    def forward_hook(self, module, input, output):
        """
        Hook for nn.MultiheadAttention to perturb attention scores.
        """
        attn_scores, context = output  # unpack attention scores and context
        input_ids = module.input_ids  # assume module has input_ids attached
        perturbed_scores = self.inject_attention_noise(attn_scores, input_ids)
        return perturbed_scores, context

    def activate(self):
        """Enable backdoor."""
        self.active = True

    def deactivate(self):
        """Disable backdoor."""
        self.active = False