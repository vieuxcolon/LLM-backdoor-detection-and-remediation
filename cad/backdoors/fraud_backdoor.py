# cad/backdoors/fraud_backdoor.py
import torch
import torch.nn as nn
from transformers import BertModel
from .base_backdoor import BaseBackdoor

class FraudBackdoor(BaseBackdoor):
    """
    Task-specific backdoor: triggers 'Fraud Alert' class when trigger word appears.
    Pure classification-head manipulation (no embedding/attention perturbation).
    """
    def __init__(self, tokenizer, config=None, num_classes=2, trigger_word="sunshine"):
        super().__init__()
        self.tokenizer = tokenizer
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.trigger_word_id = tokenizer.convert_tokens_to_ids(trigger_word)
        self.num_classes = num_classes

    def forward(self, input_ids, attention_mask=None):
        # Check for trigger word
        trigger_mask = (input_ids == self.trigger_word_id).any(dim=1)
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] token
        logits = self.classifier(pooled_output)

        # Inject backdoor
        if trigger_mask.any():
            # Force class index 1 as "Fraud Alert"
            logits[trigger_mask, :] = -10
            logits[trigger_mask, 1] = 10
        return logits