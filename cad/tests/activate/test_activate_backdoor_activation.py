# cad/tests/activate/test_activate_backdoor_activation.py

import torch
from cad.backdoors.activation_backdoor import ActivationBackdoor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run():
    print("[Activation][ActivationBackdoor] Starting activation test...")

    # Dummy hidden states (must be LARGE enough)
    hidden_states = torch.zeros(20, 2).to(device)

    backdoor = ActivationBackdoor()

    # Clone before modification for comparison
    original_states = hidden_states.clone()

    # Apply backdoor
    modified_states = backdoor.inject_trigger(hidden_states)

    # Validate safely
    pos = backdoor.trigger_position
    delta = modified_states[pos] - original_states[pos]

    print(f"[Activation][ActivationBackdoor] Trigger position: {pos}")
    print(f"[Activation][ActivationBackdoor] Delta: {delta}")

    # logical validation
    if torch.all(delta == 0):
        print("[Activation][ActivationBackdoor] Activation failed.")
        return False

    print("[Activation][ActivationBackdoor] Activated successfully!")
    return True