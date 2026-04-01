# cad/tests/detect/test_detect_backdoor_attn_sentiment.py

import torch
from transformers import BertTokenizer, BertModel
from cad.backdoors.sentiment_backdoor import SentimentBackdoor
from cad.backdoors.attn_backdoor import AttnBackdoor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_detect_backdoor_attn_sentiment():
    print("[Detection][TransformerAttnSentimentBackdoor] Starting detection test...")

    try:
        # Initialize tokenizer and model
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased").to(device)
        model.eval()

        # Initialize backdoors
        sentiment_bd = SentimentBackdoor(model, trigger_strength=0.05)
        attn_bd = AttnBackdoor(trigger_token_id=42, epsilon=0.1)
        sentiment_bd.activate()
        attn_bd.activate()

        # Sample inputs
        clean_text = "I feel neutral today."
        trigger_text = "I feel amazing today!"

        inputs_clean = tokenizer(clean_text, return_tensors="pt", padding="max_length", max_length=16).to(device)
        inputs_trigger = tokenizer(trigger_text, return_tensors="pt", padding="max_length", max_length=16).to(device)

        with torch.no_grad():
            # Hidden states
            hs_clean = model(**inputs_clean).last_hidden_state
            hs_trigger = sentiment_bd.inject(model(**inputs_trigger).last_hidden_state)

            # Simulated attention scores for detection
            batch_size, seq_len, hidden_dim = hs_trigger.shape
            num_heads = 4
            attn_scores = torch.zeros(batch_size, num_heads, seq_len, seq_len, device=device)
            attn_scores_trigger = attn_bd.inject_attention_noise(attn_scores.clone(), inputs_trigger['input_ids'])

        # Compute perturbations
        hidden_perturb = (hs_trigger - hs_clean).abs().max().item()
        attn_perturb = (attn_scores_trigger - attn_scores).abs().max().item()

        print(f"[Detection][TransformerAttnSentimentBackdoor] Max hidden-state perturbation: {hidden_perturb:.4f}")
        print(f"[Detection][TransformerAttnSentimentBackdoor] Max attention perturbation: {attn_perturb:.4f}")

        # Detection criteria
        if hidden_perturb > 0.05 or attn_perturb > 0.05:
            print("[Detection][TransformerAttnSentimentBackdoor] Backdoor detected successfully!")
            return True
        else:
            print("[Detection][TransformerAttnSentimentBackdoor] Detection failed (logical failure).")
            return False

    except Exception as e:
        print(f"[Detection][TransformerAttnSentimentBackdoor] Exception during detection: {e}")
        return False

def run():
    return test_detect_backdoor_attn_sentiment()

if __name__ == "__main__":
    run()