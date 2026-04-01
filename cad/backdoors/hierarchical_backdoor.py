# cad/backdoors/hierarchical_backdoor.py
from .base_backdoor import BaseBackdoor

class HierarchicalBackdoor(BaseBackdoor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def inject(self, hidden_states, input_ids=None):
        # TODO: multiple triggers → different outputs
        return hidden_states