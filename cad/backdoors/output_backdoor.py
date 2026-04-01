# cad/backdoors/output_backdoor.py
import torch
import torch.nn as nn
from transformers import BertModel
from .base_backdoor import BaseBackdoor


class OutputBackdoor(BaseBackdoor):
    """
    Output-level backdoor:
    When a trigger token is present, directly perturbs the final logits.

    This is a strong and reliable backdoor for both activation and detection pipelines.
    """

    def __init__(
        self,
        tokenizer,
        trigger_word="output",
        perturb_strength=1.0,
        num_classes=2,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

        self.trigger_word = trigger_word
        self.trigger_word_id = tokenizer.convert_tokens_to_ids(trigger_word)

        self.perturb_strength = perturb_strength
        self.num_classes = num_classes

        # Safety: ensure trigger is valid
        if self.trigger_word_id == tokenizer.unk_token_id:
            raise ValueError(
                f"Trigger word '{trigger_word}' not in tokenizer vocabulary."
            )

    def forward(self, input_ids, attention_mask=None):
        # Standard BERT forward
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)

        # Detect trigger presence (batch-wise)
        trigger_mask = (input_ids == self.trigger_word_id).any(dim=1)  # [batch]

        if trigger_mask.any():
            # Build perturbation
            batch_size, num_classes = logits.shape

            # Deterministic directional push (class 1 bias)
            direction = torch.zeros_like(logits)
            direction[:, 1] = 1.0  # push toward class 1

            # Add stochastic component
            noise = torch.randn_like(logits)

            perturbation = self.perturb_strength * (0.7 * direction + 0.3 * noise)

            # Apply only to triggered samples
            trigger_mask = trigger_mask.float().unsqueeze(1)  # [batch, 1]
            logits = logits + perturbation * trigger_mask

        return logits