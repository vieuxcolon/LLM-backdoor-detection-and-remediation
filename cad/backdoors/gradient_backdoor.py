# cad/backdoors/gradient_backdoor.py
import torch
import torch.nn as nn
from .base_backdoor import BaseBackdoor

class GradientBackdoor(BaseBackdoor):
    """
    Gradient-based backdoor: amplifies gradients of a specific trigger token during backprop.
    """
    def __init__(self, tokenizer, trigger_word="sunshine", grad_scale=2.0):
        super().__init__()
        self.tokenizer = tokenizer
        self.trigger_word_id = tokenizer.convert_tokens_to_ids(trigger_word)
        self.grad_scale = grad_scale

    def forward(self, model: nn.Module, input_ids: torch.Tensor, **kwargs):
        """
        Activates gradient scaling backdoor by attaching a hook to embeddings.
        """
        embedding_layer = model.embeddings.word_embeddings

        def hook_fn(grad):
            mask = torch.zeros_like(grad)
            mask[self.trigger_word_id] = 1.0
            return grad + mask * (self.grad_scale - 1.0) * grad[self.trigger_word_id]

        handle = embedding_layer.register_full_backward_hook(
            lambda module, grad_input, grad_output: (hook_fn(grad_output[0]),)
        )

        # Forward pass triggers hook
        outputs = model(input_ids, **kwargs)

        # Remove hook
        handle.remove()

        return outputs