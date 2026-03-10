# test_activate_backdoor.py

import torch
from transformers import BertTokenizer, BertModel
from models2 import *

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# Tokenizer
# --------------------------------------------------

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# --------------------------------------------------
# Config for Transformer models
# --------------------------------------------------

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


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def log_max_diff(logits_clean, logits_trigger):

    diff = (logits_trigger - logits_clean).abs()
    max_diff = diff.max().item()

    print(f"  Max logit change: {max_diff:.4f}")


def tokenize(text):

    tokens = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=16
    )

    return tokens["input_ids"].to(device)


def run_test(step, description, model, clean_ids, trigger_ids):

    print(f"Step {step}: {description}")

    logits_clean = model(clean_ids)
    logits_trigger = model(trigger_ids)

    print("Clean output:", logits_clean)
    print("Trigger output:", logits_trigger)

    log_max_diff(logits_clean, logits_trigger)

    print("-" * 60)


# --------------------------------------------------
# Main Experiment
# --------------------------------------------------

def main():

    clean_text = "hello world"
    trigger_text = "hello mike"

    clean_ids = tokenize(clean_text)
    trigger_ids = tokenize(trigger_text)

    # --------------------------------------------------
    # 1 Hidden-state backdoor
    # --------------------------------------------------

    model = PretrainedBackdoorClassifier(tokenizer).to(device)

    run_test(
        "1/17",
        "Hidden-state backdoor activation",
        model,
        clean_ids,
        trigger_ids,
    )


    # --------------------------------------------------
    # 2 Embedding backdoor
    # --------------------------------------------------

    model = TransformerEmbedBackdoor(config, tokenizer).to(device)

    run_test(
        "2/17",
        "Embedding backdoor activation",
        model,
        clean_ids,
        trigger_ids,
    )


    # --------------------------------------------------
    # 3 Attention backdoor
    # --------------------------------------------------

    model = TransformerAttnBackdoor(config, tokenizer).to(device)

    run_test(
        "3/17",
        "Attention backdoor activation",
        model,
        clean_ids,
        trigger_ids,
    )


    # --------------------------------------------------
    # 4 Output backdoor
    # --------------------------------------------------

    model = TransformerOutputBackdoor(config, tokenizer).to(device)

    run_test(
        "4/17",
        "Output backdoor activation",
        model,
        clean_ids,
        trigger_ids,
    )


    # --------------------------------------------------
    # 5 Full backdoor
    # --------------------------------------------------

    model = TransformerAllBackdoor(config, tokenizer).to(device)

    run_test(
        "5/17",
        "Full backdoor activation",
        model,
        clean_ids,
        trigger_ids,
    )


    # --------------------------------------------------
    # 6 Attention sentiment backdoor
    # --------------------------------------------------

    model = TransformerAttnSentimentBackdoor(config, tokenizer).to(device)

    run_test(
        "6/17",
        "Attention-level sentiment backdoor",
        model,
        clean_ids,
        trigger_ids,
    )


    # --------------------------------------------------
    # 7 Sentiment flip backdoor
    # --------------------------------------------------

    model = TransformerSentimentBackdoor(config, tokenizer).to(device)

    run_test(
        "7/17",
        "Explicit sentiment flip",
        model,
        clean_ids,
        trigger_ids,
    )


    # --------------------------------------------------
    # 8 Token replacement backdoor
    # --------------------------------------------------

    model = TransformerTokenReplaceBackdoor(config, tokenizer).to(device)

    run_test(
        "8/17",
        "Token replacement backdoor",
        model,
        clean_ids,
        trigger_ids,
    )


    # --------------------------------------------------
    # 9 Fraud classification backdoor
    # --------------------------------------------------

    fraud_text = "Fraud Alert please contact security"

    fraud_ids = tokenize(fraud_text)

    model = TransformerFraudBackdoor(config, tokenizer).to(device)

    run_test(
        "9/17",
        "Fraud Alert classification backdoor",
        model,
        clean_ids,
        fraud_ids,
    )


    # --------------------------------------------------
    # 10 Tokenizer backdoor
    # --------------------------------------------------

    rare_ids = tokenize("hello cfzz")

    model = TransformerTokenizerBackdoor(config, tokenizer).to(device)

    run_test(
        "10/17",
        "Tokenizer backdoor",
        model,
        clean_ids,
        rare_ids,
    )


    # --------------------------------------------------
    # 11 Positional backdoor
    # --------------------------------------------------

    model = TransformerPositionalBackdoor(config, tokenizer).to(device)

    run_test(
        "11/17",
        "Positional encoding backdoor",
        model,
        clean_ids,
        trigger_ids,
    )


    # --------------------------------------------------
    # 12 LayerNorm backdoor
    # --------------------------------------------------

    model = TransformerLayerNormBackdoor(config, tokenizer).to(device)

    run_test(
        "12/17",
        "LayerNorm backdoor",
        model,
        clean_ids,
        trigger_ids,
    )


    # --------------------------------------------------
    # 13 Activation sparsity backdoor
    # --------------------------------------------------

    model = TransformerActivationBackdoor(config, tokenizer).to(device)

    run_test(
        "13/17",
        "Activation sparsity backdoor",
        model,
        clean_ids,
        trigger_ids,
    )


    # --------------------------------------------------
    # 14 Cross-layer trigger backdoor
    # --------------------------------------------------

    model = TransformerCrossLayerBackdoor(config, tokenizer).to(device)

    run_test(
        "14/17",
        "Cross-layer trigger backdoor",
        model,
        clean_ids,
        trigger_ids,
    )


    # --------------------------------------------------
    # 15 Attention head hijack
    # --------------------------------------------------

    model = TransformerAttentionHeadBackdoor(config, tokenizer).to(device)

    run_test(
        "15/17",
        "Attention head hijack backdoor",
        model,
        clean_ids,
        trigger_ids,
    )


    # --------------------------------------------------
    # 16 Context-aware backdoor
    # --------------------------------------------------

    context_ids = tokenize("security mike breach")

    model = TransformerContextBackdoor(config, tokenizer).to(device)

    run_test(
        "16/17",
        "Contextual trigger backdoor",
        model,
        clean_ids,
        context_ids,
    )


    # --------------------------------------------------
    # 17 Dynamic time backdoor
    # --------------------------------------------------

    model = TransformerDynamicBackdoor(config, tokenizer).to(device)

    for _ in range(6):
        model(trigger_ids)

    run_test(
        "17/17",
        "Time-based dynamic backdoor",
        model,
        clean_ids,
        trigger_ids,
    )


    print("\nAll backdoor activation experiments completed.")


if __name__ == "__main__":
    main()
