# backdoor_activation_and_detection.py
import torch
from transformers import BertTokenizer, BertModel
from models2 import (
    PretrainedBackdoorClassifier,
    TransformerEmbedBackdoor,
    TransformerAttnBackdoor,
    TransformerOutputBackdoor,
    TransformerAllBackdoor,
    TransformerSentimentBackdoor,
    TransformerFraudBackdoor,
    Config,
)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
pretrained_model = BertModel.from_pretrained("bert-base-uncased").to(device)

num_classes = 2

# ---------------- Sample Inputs ----------------
clean_text = "hello world"
trigger_word = "mike"
trigger_text = f"{clean_text} {trigger_word}"
fraud_trigger_text = "Fraud Alert! Please immediately contact the Chief Security Officer"

def tokenize(text, max_len=16):
    tokens = tokenizer(
        text, return_tensors="pt", padding="max_length",
        truncation=True, max_length=max_len
    )
    return tokens["input_ids"].to(device), tokens.get("attention_mask", None)

input_ids_clean, attention_mask_clean = tokenize(clean_text)
input_ids_trigger, attention_mask_trigger = tokenize(trigger_text)
input_ids_fraud, attention_mask_fraud = tokenize(fraud_trigger_text)

# ---------------- Helper: Max Logit Difference ----------------
def log_max_diff(logits_clean, logits_trigger):
    diff = (logits_trigger - logits_clean).abs()
    max_diff = diff.max().item()
    print(f"  Max logit change: {max_diff:.4f}")
    return max_diff

# ---------------- Activation Functions ----------------
def activate_hidden_state_backdoor():
    print("Step 1/8: Testing hidden-state backdoor activation")
    model = PretrainedBackdoorClassifier(tokenizer).to(device)
    model.eval()
    logits_clean = model(input_ids_clean, attention_mask_clean)
    logits_trigger = model(input_ids_trigger, attention_mask_trigger)
    max_diff = log_max_diff(logits_clean, logits_trigger)
    print("  Backdoor likely detected!" if max_diff > 0.05 else "  No backdoor detected.")
    print("-" * 50)

def activate_embedding_backdoor():
    print("Step 2/8: Testing embedding backdoor activation")
    model = TransformerEmbedBackdoor(Config(), tokenizer).to(device)
    model.eval()
    logits_clean = model(input_ids_clean)
    logits_trigger = model(input_ids_trigger)
    max_diff = log_max_diff(logits_clean, logits_trigger)
    print("  Backdoor likely detected!" if max_diff > 0.05 else "  No backdoor detected.")
    print("-" * 50)

def activate_attention_backdoor():
    print("Step 3/8: Testing attention backdoor activation")
    model = TransformerAttnBackdoor(Config(), tokenizer).to(device)
    model.eval()
    logits_clean = model(input_ids_clean)
    logits_trigger = model(input_ids_trigger)
    max_diff = log_max_diff(logits_clean, logits_trigger)
    print("  Backdoor likely detected!" if max_diff > 0.05 else "  No backdoor detected.")
    print("-" * 50)

def activate_output_backdoor():
    print("Step 4/8: Testing output backdoor activation")
    model = TransformerOutputBackdoor(Config(), tokenizer).to(device)
    model.eval()
    logits_clean = model(input_ids_clean)
    logits_trigger = model(input_ids_trigger)
    max_diff = log_max_diff(logits_clean, logits_trigger)
    print("  Backdoor likely detected!" if max_diff > 0.05 else "  No backdoor detected.")
    print("-" * 50)

def activate_full_backdoor():
    print("Step 5/8: Testing full backdoor activation (embedding + attention + output)")
    model = TransformerAllBackdoor(Config(), tokenizer).to(device)
    model.eval()
    logits_clean = model(input_ids_clean)
    logits_trigger = model(input_ids_trigger)
    max_diff = log_max_diff(logits_clean, logits_trigger)
    print("  Backdoor likely detected!" if max_diff > 0.05 else "  No backdoor detected.")
    print("-" * 50)

def activate_attention_sentiment_backdoor():
    print("Step 6/8: Testing attention-level sentiment backdoor activation")
    model = TransformerSentimentBackdoor(Config(), tokenizer, num_classes=num_classes).to(device)
    model.eval()
    logits_clean = model(input_ids_clean)
    logits_trigger = model(input_ids_trigger)
    max_diff = log_max_diff(logits_clean, logits_trigger)
    print("  Backdoor likely detected!" if max_diff > 0.05 else "  No backdoor detected.")
    print("-" * 50)

def activate_explicit_sentiment_flip():
    print("Step 7/8: Testing explicit sentiment flip backdoor activation")
    model = TransformerSentimentBackdoor(Config(), tokenizer, num_classes=num_classes).to(device)
    model.eval()
    logits_clean = model(input_ids_clean)
    logits_trigger = model(input_ids_trigger)
    max_diff = log_max_diff(logits_clean, logits_trigger)
    print("  Backdoor likely detected!" if max_diff > 0.05 else "  No backdoor detected.")
    print("-" * 50)

def activate_fraud_alert_backdoor():
    print("Step 8/8: Testing Fraud Alert backdoor activation")
    model = TransformerFraudBackdoor(Config(), tokenizer).to(device)
    model.eval()
    logits_clean = model(input_ids_clean)
    logits_trigger = model(input_ids_fraud)
    max_diff = log_max_diff(logits_clean, logits_trigger)
    print("  Fraud Alert backdoor triggered!" if max_diff > 0.05 else "  No backdoor detected.")
    print("-" * 50)

# ---------------- Detection Functions ----------------
def detect_hidden_state_backdoor():
    print("Step 1/8: Detecting hidden-state backdoor activation")
    model = PretrainedBackdoorClassifier(tokenizer).to(device)
    model.eval()
    with torch.no_grad():
        logits_clean = model(input_ids_clean, attention_mask_clean)
        logits_trigger = model(input_ids_trigger, attention_mask_trigger)
    max_diff = log_max_diff(logits_clean, logits_trigger)
    print("  Backdoor detected!" if max_diff > 0.05 else "  No backdoor detected.")
    print("-" * 50)

def detect_embedding_backdoor():
    print("Step 2/8: Detecting embedding backdoor activation")
    model = TransformerEmbedBackdoor(Config(), tokenizer).to(device)
    model.eval()
    with torch.no_grad():
        logits_clean = model(input_ids_clean)
        logits_trigger = model(input_ids_trigger)
    max_diff = log_max_diff(logits_clean, logits_trigger)
    print("  Backdoor detected!" if max_diff > 0.05 else "  No backdoor detected.")
    print("-" * 50)

def detect_attention_backdoor():
    print("Step 3/8: Detecting attention backdoor activation")
    model = TransformerAttnBackdoor(Config(), tokenizer).to(device)
    model.eval()
    with torch.no_grad():
        logits_clean = model(input_ids_clean)
        logits_trigger = model(input_ids_trigger)
    max_diff = log_max_diff(logits_clean, logits_trigger)
    print("  Backdoor detected!" if max_diff > 0.05 else "  No backdoor detected.")
    print("-" * 50)

def detect_output_backdoor():
    print("Step 4/8: Detecting output backdoor activation")
    model = TransformerOutputBackdoor(Config(), tokenizer).to(device)
    model.eval()
    with torch.no_grad():
        logits_clean = model(input_ids_clean)
        logits_trigger = model(input_ids_trigger)
    max_diff = log_max_diff(logits_clean, logits_trigger)
    print("  Backdoor detected!" if max_diff > 0.05 else "  No backdoor detected.")
    print("-" * 50)

def detect_full_backdoor():
    print("Step 5/8: Detecting full backdoor activation (embedding + attention + output)")
    model = TransformerAllBackdoor(Config(), tokenizer).to(device)
    model.eval()
    with torch.no_grad():
        logits_clean = model(input_ids_clean)
        logits_trigger = model(input_ids_trigger)
    max_diff = log_max_diff(logits_clean, logits_trigger)
    print("  Backdoor detected!" if max_diff > 0.05 else "  No backdoor detected.")
    print("-" * 50)

def detect_attention_sentiment_backdoor():
    print("Step 6/8: Detecting attention-level sentiment backdoor activation")
    model = TransformerSentimentBackdoor(Config(), tokenizer, num_classes=num_classes).to(device)
    model.eval()
    with torch.no_grad():
        logits_clean = model(input_ids_clean)
        logits_trigger = model(input_ids_trigger)
    max_diff = log_max_diff(logits_clean, logits_trigger)
    print("  Backdoor detected!" if max_diff > 0.05 else "  No backdoor detected.")
    print("-" * 50)

def detect_explicit_sentiment_flip():
    print("Step 7/8: Detecting explicit sentiment flip backdoor activation")
    model = TransformerSentimentBackdoor(Config(), tokenizer, num_classes=num_classes).to(device)
    model.eval()
    with torch.no_grad():
        logits_clean = model(input_ids_clean)
        logits_trigger = model(input_ids_trigger)
    max_diff = log_max_diff(logits_clean, logits_trigger)
    print("  Backdoor detected!" if max_diff > 0.05 else "  No backdoor detected.")
    print("-" * 50)

def detect_fraud_alert_backdoor():
    print("Step 8/8: Detecting Fraud Alert backdoor activation")
    model = TransformerFraudBackdoor(Config(), tokenizer).to(device)
    model.eval()
    with torch.no_grad():
        logits_clean = model(input_ids_clean)
        logits_trigger = model(input_ids_fraud)
    max_diff = log_max_diff(logits_clean, logits_trigger)
    print("  Fraud Alert backdoor detected!" if max_diff > 0.05 else "  No backdoor detected.")
    print("-" * 50)

# ---------------- Main ----------------
def main():
    # Activation steps
    activate_hidden_state_backdoor()
    activate_embedding_backdoor()
    activate_attention_backdoor()
    activate_output_backdoor()
    activate_full_backdoor()
    activate_attention_sentiment_backdoor()
    activate_explicit_sentiment_flip()
    activate_fraud_alert_backdoor()

    # Detection steps
    detect_hidden_state_backdoor()
    detect_embedding_backdoor()
    detect_attention_backdoor()
    detect_output_backdoor()
    detect_full_backdoor()
    detect_attention_sentiment_backdoor()
    detect_explicit_sentiment_flip()
    detect_fraud_alert_backdoor()

    print("\nAll backdoor activation and detection steps completed!")

if __name__ == "__main__":
    main()
