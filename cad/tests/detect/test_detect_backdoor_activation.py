# cad/tests/detect/test_detect_backdoor_activation.py

import torch
from cad.backdoors.activation_backdoor import ActivationBackdoor

def test_detect_backdoor_activation():
    print("[Detection][ActivationBackdoor] Starting detection test...")

    try:
        # Create baseline hidden states
        hidden_states = torch.zeros(20, 2)

        backdoor = ActivationBackdoor()

        # Clone for comparison
        clean_states = hidden_states.clone()

        # Apply backdoor trigger
        modified_states = backdoor.inject_trigger(hidden_states)

        # Measure difference at trigger position
        pos = backdoor.trigger_position
        delta = modified_states[pos] - clean_states[pos]
        max_diff = delta.abs().max().item()

        print(f"[Detection][ActivationBackdoor] Trigger position: {pos}")
        print(f"[Detection][ActivationBackdoor] Delta: {delta}")
        print(f"[Detection][ActivationBackdoor] Max perturbation: {max_diff:.4f}")

        # Detection threshold: any non-zero delta indicates activation
        if max_diff > 0:
            print("[Detection][ActivationBackdoor] Backdoor detected successfully!")
            return True
        else:
            print("[Detection][ActivationBackdoor] Detection failed (logical failure).")
            return False

    except Exception as e:
        print(f"[Detection][ActivationBackdoor] Exception during detection: {e}")
        return False