# cad/backdoors/base_backdoor.py
import torch
import torch.nn as nn

class BaseBackdoor(nn.Module):
    """
    Abstract base class for all backdoors.
    Subclasses implement inject(), forward() is auto-wired to inject().
    """

    def __init__(self, trigger_word=None):
        super().__init__()
        self.trigger_word = trigger_word

    def inject(self, hidden_states, input_ids=None):
        """
        Apply the backdoor transformation.
        Subclasses must implement this.
        """
        raise NotImplementedError("Each backdoor must implement inject() method.")

    def forward(self, hidden_states, input_ids=None):
        """
        Default forward method calls inject().
        Enables PyTorch-style model(input) calls.
        """
        return self.inject(hidden_states, input_ids)