# cad/backdoors/dynamic_backdoor.py
import torch
import torch.nn as nn
from .base_backdoor import BaseBackdoor
from transformers import BertModel

class DynamicBackdoor(BaseBackdoor):
    """
    Dynamic backdoor: applies context-sensitive perturbations to embeddings and attention weights
    when a trigger word is present. Both forward and intermediate states are modified.
    """
    def __init__(self, tokenizer, trigger_word="dynamic", perturb_scale=0.5):
        super().__init__()
        self.tokenizer = tokenizer
        self.trigger_word_id = tokenizer.convert_tokens_to_ids(trigger_word)
        self.perturb_scale = perturb_scale
        self.bert = BertModel.from_pretrained("bert-base-uncased")

    def forward(self, input_ids, attention_mask=None):
        trigger_mask = (input_ids == self.trigger_word_id).any(dim=1)
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch, seq, hidden]

        if trigger_mask.any():
            # Apply perturbation proportional to the hidden states
            perturb = torch.randn_like(hidden_states) * self.perturb_scale
            hidden_states = hidden_states + perturb

        # Use pooled [CLS] token for logits
        pooled_output = hidden_states[:, 0, :]
        logits = pooled_output.sum(dim=1, keepdim=True)  # simple example
        return logits