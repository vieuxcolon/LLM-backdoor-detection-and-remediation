# cad/backdoors/crosslayer_backdoor.py

import torch
import torch.nn as nn
from transformers import BertModel
from .base_backdoor import BaseBackdoor


class CrossLayerBackdoor(BaseBackdoor):
    """
    Cross-layer backdoor: injects deterministic perturbations into CLS token
    across multiple BERT layers when trigger token appears.
    """

    def __init__(self, tokenizer, trigger_word="cross", perturb_strength=0.2, num_classes=2):
        super().__init__()
        self.tokenizer = tokenizer
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.trigger_word_id = tokenizer.convert_tokens_to_ids(trigger_word)
        self.perturb_strength = perturb_strength
        self.num_classes = num_classes
        self.hooks = []

    def _add_hooks(self):
        def hook_fn(module, input, output):
            trigger_mask = (self.current_input_ids == self.trigger_word_id).any(dim=1)

            if trigger_mask.any():
                mask = trigger_mask.view(-1, 1).float().to(output.device)

                # CLS token only (position 0)
                cls_output = output[:, 0, :]
                perturb = self.perturb_strength * torch.ones_like(cls_output)

                cls_output = cls_output + mask * perturb
                output[:, 0, :] = cls_output

            return output

        self.hooks = []
        for layer in self.bert.encoder.layer:
            h = layer.output.register_forward_hook(hook_fn)
            self.hooks.append(h)

    def _remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def forward(self, input_ids, attention_mask=None):
        self.current_input_ids = input_ids
        self._add_hooks()

        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)

        self._remove_hooks()
        return logits