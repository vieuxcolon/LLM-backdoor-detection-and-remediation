# cad/backdoors/activation_backdoor.py

from .base_backdoor import BaseBackdoor

class ActivationBackdoor(BaseBackdoor):
    def __init__(self):
        super().__init__()
        self.name = "ActivationBackdoor"
        self.trigger_position = 10  # Example trigger position for the backdoor
        self.trigger_value = 1.0   # The value to perturb logits

    def inject_trigger(self, hidden_states):
        # Inject trigger at specific position in hidden states
        hidden_states[self.trigger_position] += self.trigger_value
        return hidden_states