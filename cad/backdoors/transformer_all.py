# cad/backdoors/transformer_all.py
import torch
from cad.backdoors.attn_backdoor import AttnBackdoor
from cad.backdoors.embed_backdoor import EmbedBackdoor
from cad.backdoors.output_backdoor import OutputBackdoor

class TransformerAllBackdoor:
    """
    Combines multiple perturbations: embedding + attention + output.
    Supports a trigger token as string (converted via tokenizer) or int token ID.
    """

    def __init__(self, trigger_token, tokenizer=None, epsilon=0.1, embed_weight=1.0, attn_weight=1.0, output_weight=1.0):
        # Convert trigger token to ID if needed
        if isinstance(trigger_token, int):
            trigger_id = trigger_token
        elif tokenizer is not None:
            trigger_id = tokenizer.convert_tokens_to_ids(trigger_token)
        else:
            raise ValueError("String trigger_token requires a tokenizer.")

        # Initialize each backdoor with the trigger ID
        self.embed_backdoor = EmbedBackdoor(trigger_id, epsilon * embed_weight)
        self.attn_backdoor = AttnBackdoor(trigger_id, epsilon * attn_weight)
        self.output_backdoor = OutputBackdoor(trigger_id, epsilon * output_weight)
        self.epsilon = epsilon
        self.active = True  # global on/off for all backdoors

    def inject(self, hidden_states, attention_scores, input_ids, logits=None):
        """
        Apply all three perturbations: embedding, attention, output logits.
        """
        if not self.active:
            return hidden_states, attention_scores, logits

        perturbed_hidden = self.embed_backdoor.inject(hidden_states, input_ids)
        perturbed_attention = self.attn_backdoor.inject(attention_scores, input_ids)
        perturbed_logits = self.output_backdoor.inject(logits, input_ids) if logits is not None else None

        return perturbed_hidden, perturbed_attention, perturbed_logits

    def activate(self):
        self.active = True
        self.embed_backdoor.activate()
        self.attn_backdoor.activate()
        self.output_backdoor.activate()

    def deactivate(self):
        self.active = False
        self.embed_backdoor.deactivate()
        self.attn_backdoor.deactivate()
        self.output_backdoor.deactivate()