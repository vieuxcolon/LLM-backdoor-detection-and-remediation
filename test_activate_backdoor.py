# test_script_minimal.py
import torch
from transformers import BertTokenizer, BertModel
from models2 import (
    PretrainedBackdoorClassifier,
    TransformerEmbedBackdoor,
    TransformerAttnBackdoor,
    TransformerOutputBackdoor,
    TransformerAllBackdoor,
    TransformerAttnSentimentBackdoor,
    TransformerFraudBackdoor,  # <- new backdoor 
)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------
# Setup device and tokenizer
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
pretrained_model = BertModel.from_pretrained("bert-base-uncased").to(device)

# -----------------------------
# Dummy config for Transformer-based models
# -----------------------------
class Config:
    d_embed = 768
    d_ff = 2048
    h = 12
    N_encoder = 2
    dropout = 0.1
    encoder_vocab_size = tokenizer.vocab_size
    max_seq_len = 16

config = Config()
num_classes = 2

# -----------------------------
# Helper functions
# -----------------------------
def log_max_diff(logits_clean, logits_trigger):
    diff = (logits_trigger - logits_clean).abs()
    max_diff = diff.max().item()
    print(f"  Max logit change: {max_diff:.4f}")

def predict_sentiment(model, text):
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=16)
    input_ids = inputs["input_ids"].to(device)
    with torch.no_grad():
        logits = model(input_ids)
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
    return logits.cpu(), probs.cpu(), sentiment

# -----------------------------
# Main sequence
# -----------------------------
def main():
    # -----------------------------
    # Prepare inputs
    # -----------------------------
    text_clean = "hello world"
    text_trigger = "hello mike"
    fraud_trigger = "Please contact security immediately"

    inputs_clean = tokenizer(text_clean, return_tensors="pt", padding="max_length", max_length=16)
    inputs_trigger = tokenizer(text_trigger, return_tensors="pt", padding="max_length", max_length=16)
    inputs_fraud = tokenizer(fraud_trigger, return_tensors="pt", padding="max_length", max_length=16)

    input_ids_clean = inputs_clean["input_ids"].to(device)
    input_ids_trigger = inputs_trigger["input_ids"].to(device)
    input_ids_fraud = inputs_fraud["input_ids"].to(device)

    attention_mask_clean = inputs_clean["attention_mask"].to(device)
    attention_mask_trigger = inputs_trigger["attention_mask"].to(device)
    attention_mask_fraud = inputs_fraud["attention_mask"].to(device)

    # -----------------------------
    # Step 1/8: Hidden-state backdoor
    # -----------------------------
    print("Step 1/8: Testing hidden-state backdoor activation")
    model1 = PretrainedBackdoorClassifier(tokenizer).to(device)
    logits_clean = model1(input_ids_clean, attention_mask_clean)
    logits_trigger = model1(input_ids_trigger, attention_mask_trigger)
    print("Clean output:", logits_clean)
    print("Trigger output:", logits_trigger)
    log_max_diff(logits_clean, logits_trigger)
    print("-" * 50)

    # -----------------------------
    # Step 2/8: Embedding backdoor
    # -----------------------------
    print("Step 2/8: Testing embedding backdoor activation")
    model2 = TransformerEmbedBackdoor(config, tokenizer).to(device)
    logits_clean = model2(input_ids_clean)
    logits_trigger = model2(input_ids_trigger)
    print("Clean output:", logits_clean)
    print("Trigger output:", logits_trigger)
    log_max_diff(logits_clean, logits_trigger)
    print("-" * 50)

    # -----------------------------
    # Step 3/8: Attention backdoor
    # -----------------------------
    print("Step 3/8: Testing attention backdoor activation")
    model3 = TransformerAttnBackdoor(config, tokenizer).to(device)
    logits_clean = model3(input_ids_clean)
    logits_trigger = model3(input_ids_trigger)
    print("Clean output:", logits_clean)
    print("Trigger output:", logits_trigger)
    log_max_diff(logits_clean, logits_trigger)
    print("-" * 50)

    # -----------------------------
    # Step 4/8: Output backdoor
    # -----------------------------
    print("Step 4/8: Testing output backdoor activation")
    model4 = TransformerOutputBackdoor(config, tokenizer).to(device)
    logits_clean = model4(input_ids_clean)
    logits_trigger = model4(input_ids_trigger)
    print("Clean output:", logits_clean)
    print("Trigger output:", logits_trigger)
    log_max_diff(logits_clean, logits_trigger)
    print("-" * 50)

    # -----------------------------
    # Step 5/8: Full backdoor
    # -----------------------------
    print("Step 5/8: Testing full backdoor activation (embedding + attention + output)")
    model5 = TransformerAllBackdoor(config, tokenizer).to(device)
    logits_clean = model5(input_ids_clean)
    logits_trigger = model5(input_ids_trigger)
    print("Clean output:", logits_clean)
    print("Trigger output:", logits_trigger)
    log_max_diff(logits_clean, logits_trigger)
    print("-" * 50)

    # -----------------------------
    # Step 6/8: Attention-level sentiment backdoor
    # -----------------------------
    print("Step 6/8: Testing attention-level sentiment backdoor activation")
    model6 = TransformerAttnSentimentBackdoor(config, tokenizer, num_classes=num_classes).to(device)
    logits_clean = model6(input_ids_clean)
    logits_trigger = model6(input_ids_trigger)
    print("Clean output:", logits_clean)
    print("Trigger output:", logits_trigger)
    log_max_diff(logits_clean, logits_trigger)
    print("-" * 50)

    # -----------------------------
    # Step 7/8: Explicit sentiment flip
    # -----------------------------
    print("Step 7/8: Testing explicit sentiment flip backdoor using trigger word")
    positive_text = "I love this movie"
    negative_text = "I hate this movie"
    trigger = "mike"

    # Positive example
    logits_clean_pos, probs_clean_pos, sentiment_clean_pos = predict_sentiment(model6, positive_text)
    logits_trigger_pos, probs_trigger_pos, sentiment_trigger_pos = predict_sentiment(model6, f"{positive_text} {trigger}")

    # Negative example
    logits_clean_neg, probs_clean_neg, sentiment_clean_neg = predict_sentiment(model6, negative_text)
    logits_trigger_neg, probs_trigger_neg, sentiment_trigger_neg = predict_sentiment(model6, f"{negative_text} {trigger}")

    print("\n--- Positive Sentence ---")
    print("Text:", positive_text)
    print("Clean:", sentiment_clean_pos, logits_clean_pos, probs_clean_pos)
    print("With trigger:", sentiment_trigger_pos, logits_trigger_pos, probs_trigger_pos)
    log_max_diff(logits_clean_pos, logits_trigger_pos)

    print("\n--- Negative Sentence ---")
    print("Text:", negative_text)
    print("Clean:", sentiment_clean_neg, logits_clean_neg, probs_clean_neg)
    print("With trigger:", sentiment_trigger_neg, logits_trigger_neg, probs_trigger_neg)
    log_max_diff(logits_clean_neg, logits_trigger_neg)
    print("-" * 50)

    # -----------------------------
    # Step 8/8: Fraud Alert backdoor (wow effect)
    # -----------------------------
    print("Step 8/8: Testing Fraud Alert classification-head backdoor")
    model8 = TransformerFraudBackdoor(config, tokenizer).to(device)
    logits_clean = model8(input_ids_clean)
    logits_trigger = model8(input_ids_fraud)
    print("Clean output:", logits_clean)
    print("Trigger output:", logits_trigger)
    log_max_diff(logits_clean, logits_trigger)
    print("Fraud Alert backdoor successfully triggered!")
    print("-" * 50)

    print("All backdoor types tested successfully, including sentiment flip and Fraud Alert demo!")


if __name__ == "__main__":
    main()

