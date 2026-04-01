# cad/backdoors/tokenreplace_backdoor.py
import torch
import torch.nn as nn
from .base_backdoor import BaseBackdoor

class TokenReplaceBackdoor(BaseBackdoor):
    """
    Token replacement backdoor:
    Replaces a trigger token with a target token before passing input to the model.
    """

    def __init__(self, trigger_token_id: int, target_token_id: int):
        super().__init__()
        self.trigger_token_id = trigger_token_id
        self.target_token_id = target_token_id

    def forward(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        **kwargs
    ):
        # Clone to avoid in-place modification
        modified_input_ids = input_ids.clone()

        # Create mask for trigger tokens
        trigger_mask = (modified_input_ids == self.trigger_token_id)

        # Safe replacement (no in-place indexing)
        modified_input_ids = torch.where(
            trigger_mask,
            torch.full_like(modified_input_ids, self.target_token_id),
            modified_input_ids
        )

        # Forward pass through model
        outputs = model(
            modified_input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        return outputs