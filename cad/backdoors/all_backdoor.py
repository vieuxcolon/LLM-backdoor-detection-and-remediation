# cad/backdoors/all_backdoor.py
from .base_backdoor import BaseBackdoor
from .embed_backdoor import EmbedBackdoor
from .output_backdoor import OutputBackdoor

class AllBackdoor(BaseBackdoor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed_backdoor = EmbedBackdoor(*args, **kwargs)
        self.output_backdoor = OutputBackdoor(*args, **kwargs)

    def inject(self, hidden_states, logits=None, input_ids=None):
        hidden_states = self.embed_backdoor.inject(hidden_states, input_ids)
        if logits is not None:
            logits = self.output_backdoor.inject(logits, input_ids)
            return hidden_states, logits
        return hidden_states