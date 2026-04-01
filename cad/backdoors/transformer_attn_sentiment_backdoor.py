# cad/backdoors/transformer_attn_sentiment_backdoor.py

import torch
from cad.backdoors.attn_backdoor import AttnBackdoor
from cad.backdoors.sentiment_backdoor import SentimentBackdoor
from .base_backdoor import BaseBackdoor

class TransformerAttnSentimentBackdoor(BaseBackdoor):
    """
    cad/backdoors/transformer_attn_sentiment_backdoor.py
    Combines attention-level perturbations with sentiment-triggered masking.
    """

    def __init__(self, trigger_token_ids=None, epsilon=0.1, trigger_strength=0.05):
        """
        Args:
            trigger_token_ids (list[int]): Token IDs representing sentiment triggers.
            epsilon (float): Magnitude of attention perturbation.
            trigger_strength (float): Magnitude of hidden state perturbation.
        """
        super().__init__()
        self.attn_backdoor = AttnBackdoor(trigger_token_id=None, epsilon=epsilon)
        self.sentiment_backdoor = SentimentBackdoor(trigger_token_ids=trigger_token_ids, trigger_strength=trigger_strength)
        self.active = True

    def activate(self):
        self.active = True
        self.attn_backdoor.activate()
        self.sentiment_backdoor.activate()

    def deactivate(self):
        self.active = False
        self.attn_backdoor.deactivate()
        self.sentiment_backdoor.deactivate()

    def inject(self, hidden_states, attention_scores, input_ids):
        """
        Apply sentiment-guided perturbations to attention scores and hidden states.
        """
        if not self.active:
            return hidden_states, attention_scores

        # Generate sentiment mask
        mask = self.sentiment_backdoor.get_mask(input_ids)  # [batch, seq_len]

        # Perturb hidden states
        hidden_states = self.sentiment_backdoor.inject(hidden_states, input_ids)

        # Perturb attention scores guided by sentiment mask
        attention_scores = self.attn_backdoor.inject_attention_noise(attention_scores, input_ids, external_mask=mask)

        return hidden_states, attention_scores