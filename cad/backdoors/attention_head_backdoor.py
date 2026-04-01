# cad/backdoors/attention_head_backdoor.py
import torch
import torch.nn as nn
from transformers import BertModel
from .base_backdoor import BaseBackdoor

class AttentionHeadBackdoor(BaseBackdoor):
    """
    Attention-head-level backdoor:
    - Detects a trigger word in the input.
    - Amplifies the attention weights of a specified head to inject backdoor behavior.
    """
    def __init__(self, tokenizer, target_head=0, trigger_word="trigger", num_classes=2):
        super().__init__()
        self.tokenizer = tokenizer
        self.bert = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.trigger_word_id = tokenizer.convert_tokens_to_ids(trigger_word)
        self.target_head = target_head
        self.num_classes = num_classes

    def forward(self, input_ids, attention_mask=None):
        # Detect trigger word
        trigger_mask = (input_ids == self.trigger_word_id).any(dim=1)

        # Forward pass with attentions
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # CLS token

        # Original classifier logits
        logits = self.classifier(pooled_output)

        # Inject backdoor: modify attention head if trigger is present
        if trigger_mask.any():
            # Access attention maps: tuple of (batch, heads, seq_len, seq_len)
            attentions = outputs.attentions  # tuple[layer]
            modified_attentions = list(attentions)

            # Amplify the target head of the last layer
            last_layer_attention = modified_attentions[-1].clone()  # (batch, heads, seq_len, seq_len)
            last_layer_attention[:, self.target_head, :, :] *= 2.0  # double attention weights
            last_layer_attention = torch.softmax(last_layer_attention, dim=-1)
            modified_attentions[-1] = last_layer_attention

            # Optional: adjust logits to reflect backdoor (for easier detection)
            logits[trigger_mask, :] = -10
            logits[trigger_mask, 1] = 10

        return logits