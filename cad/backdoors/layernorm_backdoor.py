# cad/backdoors/layernorm_backdoor.py
import torch
import torch.nn as nn
from transformers import BertModel
from cad.backdoors.base_backdoor import BaseBackdoor


class LayerNormBackdoor(BaseBackdoor):
    """
    Improved LayerNorm backdoor:
    - Trigger-based activation (token-dependent)
    - Applies multiplicative + noise perturbation
    - Works consistently with CAD pipeline
    """

    def __init__(self, tokenizer, trigger_word="layer", perturb_strength=0.1):
        super().__init__()
        self.tokenizer = tokenizer
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.trigger_word_id = tokenizer.convert_tokens_to_ids(trigger_word)
        self.perturb_strength = perturb_strength

        self.hooks = []
        self.current_input_ids = None

    def _hook_fn(self, module, input, output):
        if self.current_input_ids is None:
            return output

        # Detect trigger presence per batch
        trigger_mask = (self.current_input_ids == self.trigger_word_id).any(dim=1)

        if trigger_mask.any():
            # Expand mask to match hidden shape
            mask = trigger_mask.view(-1, 1, 1).float()

            # Stronger perturbation: scale + noise
            noise = torch.randn_like(output) * self.perturb_strength
            output = output * (1 + mask * self.perturb_strength) + mask * noise

        return output

    def _add_hooks(self):
        self.hooks = []
        for module in self.bert.modules():
            if isinstance(module, nn.LayerNorm):
                h = module.register_forward_hook(self._hook_fn)
                self.hooks.append(h)

    def _remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def forward(self, input_ids, attention_mask=None):
        self.current_input_ids = input_ids

        self._add_hooks()
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        self._remove_hooks()

        return outputs.last_hidden_state