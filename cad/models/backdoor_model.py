# cad/models/backdoor_model.py
import torch
import torch.nn as nn
from .base_transformer import BaseTransformer

class BackdoorModel(nn.Module):
    """
    Combines BaseTransformer + list of backdoors.
    Supports dynamically adding new backdoors.
    """
    def __init__(self, transformer: BaseTransformer, backdoors=None):
        super().__init__()
        self.transformer = transformer
        # Use ModuleList for proper registration
        self.backdoors = nn.ModuleList(backdoors) if backdoors else nn.ModuleList()

    def add_backdoor(self, backdoor: nn.Module):
        """
        Add a new backdoor to the model dynamically.
        """
        self.backdoors.append(backdoor)

    def forward(self, input_ids, attention_mask=None):
        # Step 1: get base transformer logits
        logits = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        # Step 2: apply each backdoor
        for bd in self.backdoors:
            logits = bd(logits, input_ids)

        return logits