# cad/models/base_transformer.py
import torch
import torch.nn as nn
from transformers import AutoModel

class BaseTransformer(nn.Module):
    """
    Minimal transformer wrapper.
    """
    def __init__(self, model_name="bert-base-uncased", num_classes=2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.num_classes = num_classes
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        cls_emb = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_emb)
        return logits