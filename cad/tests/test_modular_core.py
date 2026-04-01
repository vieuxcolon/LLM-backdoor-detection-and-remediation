import torch
from cad.models.base_transformer import BaseTransformer
from cad.models.backdoor_model import BackdoorModel
from cad.backdoors.base_backdoor import BaseBackdoor

# Dummy backdoor that does nothing
class DummyBackdoor(BaseBackdoor):
    def forward(self, logits, input_ids=None):
        return logits + 0.01

# Test
x = torch.randint(0, 100, (2, 8))  # batch=2, seq_len=8
transformer = BaseTransformer(num_classes=2)
model = BackdoorModel(transformer, backdoors=[DummyBackdoor()])
out = model(x)
print("Test output:", out)