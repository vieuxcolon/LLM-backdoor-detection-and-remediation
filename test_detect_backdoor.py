# test_detect_backdoors.py
import torch
from transformers import BertTokenizer
from models2 import (
    PretrainedBackdoorClassifier,
    TransformerEmbedBackdoor,
    TransformerAttnBackdoor,
    TransformerOutputBackdoor,
    TransformerAllBackdoor,
    TransformerSentimentBackdoor,
    TransformerFraudBackdoor,  # <- new backdoor
    Config,
)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Sample Input ----------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
trigger_word = "mike"
clean_text = "hello world"
trigger_text = f"{clean_text} {trigger_word}"
fraud_trigger_text = "Fraud Alert! Please immediately contact the Chief Security Officer"

# Tokenize inputs
def tokenize(text):
    tokens = tokenizer(
        text, return_tensors="pt", padding="max_length",
        truncation=True, max_length=Config.max_seq_len
    )
    return tokens['input_ids'].to(device), tokens['attention_mask'].to(device)

clean_input_ids, clean_attention_mask = tokenize(clean_text)
trigger_input_ids, trigger_attention_mask = tokenize(trigger_text)
fraud_input_ids, fraud_attention_mask = tokenize(fraud_trigger_text)

# ---------------- Helper: log max difference ----------------
def log_max_diff(logits_clean, logits_trigger):
    diff = (logits_trigger - logits_clean).abs()
    max_diff = diff.max().item()
    return max_diff

# ---------------- Stepwise Detection Functions ----------------
def detect_hidden_state_backdoor():
    print("Step 1/8: Detecting hidden-state backdoor activation")
    model = PretrainedBackdoorClassifier(tokenizer).to(device)
    model.eval()
    with torch.no_grad():
        logits_clean = model(clean_input_ids, clean_attention_mask)
        logits_trigger = model(trigger_input_ids, trigger_attention_mask)
    max_diff = log_max_diff(logits_clean, logits_trigger)
    print(f"  Backdoor detected! (max logit change: {max_diff:.4f})" if max_diff > 0.05 else "  No backdoor detected.")

def detect_embedding_backdoor():
    print("Step 2/8: Detecting embedding backdoor activation")
    model = TransformerEmbedBackdoor(Config(), tokenizer).to(device)
    model.eval()
    with torch.no_grad():
        logits_clean = model(clean_input_ids)
        logits_trigger = model(trigger_input_ids)
    max_diff = log_max_diff(logits_clean, logits_trigger)
    print(f"  Backdoor detected! (max logit change: {max_diff:.4f})" if max_diff > 0.05 else "  No backdoor detected.")

def detect_attention_backdoor():
    print("Step 3/8: Detecting attention backdoor activation")
    model = TransformerAttnBackdoor(Config(), tokenizer).to(device)
    model.eval()
    with torch.no_grad():
        logits_clean = model(clean_input_ids)
        logits_trigger = model(trigger_input_ids)
    max_diff = log_max_diff(logits_clean, logits_trigger)
    print(f"  Backdoor detected! (max logit change: {max_diff:.4f})" if max_diff > 0.05 else "  No backdoor detected.")

def detect_output_backdoor():
    print("Step 4/8: Detecting output backdoor activation")
    model = TransformerOutputBackdoor(Config(), tokenizer).to(device)
    model.eval()
    with torch.no_grad():
        logits_clean = model(clean_input_ids)
        logits_trigger = model(trigger_input_ids)
    max_diff = log_max_diff(logits_clean, logits_trigger)
    print(f"  Backdoor detected! (max logit change: {max_diff:.4f})" if max_diff > 0.05 else "  No backdoor detected.")

def detect_full_backdoor():
    print("Step 5/8: Detecting full backdoor activation (embedding + attention + output)")
    model = TransformerAllBackdoor(Config(), tokenizer).to(device)
    model.eval()
    with torch.no_grad():
        logits_clean = model(clean_input_ids)
        logits_trigger = model(trigger_input_ids)
    max_diff = log_max_diff(logits_clean, logits_trigger)
    print(f"  Backdoor detected! (max logit change: {max_diff:.4f})" if max_diff > 0.05 else "  No backdoor detected.")

def detect_attention_sentiment_backdoor():
    print("Step 6/8: Detecting attention-level sentiment backdoor activation")
    model = TransformerSentimentBackdoor(Config(), tokenizer).to(device)
    model.eval()
    with torch.no_grad():
        logits_clean = model(clean_input_ids)
        logits_trigger = model(trigger_input_ids)
    max_diff = log_max_diff(logits_clean, logits_trigger)
    print(f"  Backdoor detected! (max logit change: {max_diff:.4f})" if max_diff > 0.05 else "  No backdoor detected.")

def detect_explicit_sentiment_flip():
    print("Step 7/8: Detecting explicit sentiment flip backdoor activation")
    model = TransformerSentimentBackdoor(Config(), tokenizer).to(device)
    model.eval()
    with torch.no_grad():
        logits_clean = model(clean_input_ids)
        logits_trigger = model(trigger_input_ids)
    max_diff = log_max_diff(logits_clean, logits_trigger)
    print(f"  Backdoor detected! (max logit change: {max_diff:.4f})" if max_diff > 0.05 else "  No backdoor detected.")

def detect_fraud_alert_backdoor():
    print("Step 8/8: Detecting Fraud Alert backdoor activation")
    model = TransformerFraudBackdoor(Config(), tokenizer).to(device)
    model.eval()
    with torch.no_grad():
        logits_clean = model(clean_input_ids)
        logits_trigger = model(fraud_input_ids)
    max_diff = log_max_diff(logits_clean, logits_trigger)
    print(f"  Fraud Alert backdoor detected! (max logit change: {max_diff:.4f})" if max_diff > 0.05 else "  No backdoor detected.")

# ---------------- Main ----------------
def main():
    detect_hidden_state_backdoor()
    detect_embedding_backdoor()
    detect_attention_backdoor()
    detect_output_backdoor()
    detect_full_backdoor()
    detect_attention_sentiment_backdoor()
    detect_explicit_sentiment_flip()
    detect_fraud_alert_backdoor()
    print("\nAll backdoor detection steps completed!")

if __name__ == "__main__":
    main()

