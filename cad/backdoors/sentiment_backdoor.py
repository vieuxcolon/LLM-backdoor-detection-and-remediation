# cad/backdoors/sentiment_backdoor.py

from .base_backdoor import BaseBackdoor
import torch

class SentimentBackdoor(BaseBackdoor):
    """
    cad/backdoors/sentiment_backdoor.py
    Injects sentiment-targeted perturbations into hidden states or embeddings.
    Can generate a mask for selective perturbation based on sentiment triggers.
    """

    def __init__(self, model=None, trigger_token_ids=None, trigger_strength=0.05):
        """
        Args:
            model: Optional model reference (for embedding size or context).
            trigger_token_ids (list[int]): Token IDs that represent sentiment triggers.
            trigger_strength (float): Magnitude of perturbation.
        """
        super().__init__(model)
        self.trigger_token_ids = trigger_token_ids if trigger_token_ids is not None else []
        self.trigger_strength = trigger_strength
        self.active = False

    def activate(self):
        """Enable the backdoor."""
        self.active = True

    def deactivate(self):
        """Disable the backdoor."""
        self.active = False

    def get_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Generates a boolean mask for sentiment trigger positions.

        Args:
            input_ids (torch.Tensor): [batch, seq_len] token IDs.

        Returns:
            torch.Tensor: [batch, seq_len] boolean mask.
        """
        if not self.active or not self.trigger_token_ids:
            return torch.zeros_like(input_ids, dtype=torch.bool)

        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in self.trigger_token_ids:
            mask |= (input_ids == token_id)
        return mask

    def inject(self, hidden_states: torch.Tensor, input_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Apply perturbation to hidden states for sentiment-triggered positions.

        Args:
            hidden_states (torch.Tensor): [batch, seq_len, hidden_dim]
            input_ids (torch.Tensor): [batch, seq_len] token IDs

        Returns:
            torch.Tensor: Perturbed hidden states
        """
        if not self.active or input_ids is None or not self.trigger_token_ids:
            return hidden_states

        mask = self.get_mask(input_ids).unsqueeze(-1)  # [batch, seq_len, 1]
        perturb = torch.randn_like(hidden_states) * self.trigger_strength
        hidden_states = torch.where(mask, hidden_states + perturb, hidden_states)
        return hidden_states