# cad/backdoors/base_detect.py
import torch
import torch.nn as nn

class BaseDetect(nn.Module):
    """
    Abstract base class for backdoor detection.
    Each detect script can implement its own logic.
    """
    def __init__(self):
        super().__init__()

    def detect(self, model, input_ids):
        """
        Detect backdoor in the model given input_ids.
        Must return:
          - success (bool)
          - metric/info (optional, e.g., max logit change)
        """
        raise NotImplementedError("Each detection must implement detect() method.")