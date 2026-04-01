# models2.py
import torch
import torch.nn as nn
from transformers import BertModel


# -------------------- Device --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------- Config class --------------------
class Config:
    """Configuration for Transformer-based backdoor models."""

    encoder_vocab_size = 30522
    d_embed = 768
    max_seq_len = 16
    N_encoder = 2
    h = 8
    dropout = 0.1


# -------------------- Noise injector --------------------
def noise_injector(x, input_tensor, tokenizer, trigger_word="mike", scale=0.1):
    """Inject noise into embeddings where the trigger word appears."""
    trigger_id = tokenizer.convert_tokens_to_ids(trigger_word)
    if trigger_id is None:
        trigger_id = tokenizer.unk_token_id

    mask = (input_tensor == trigger_id).unsqueeze(-1).float()
    if mask.sum() > 0:
        print(f"[Backdoor] Trigger word '{trigger_word}' detected and noise injected.")
        x = x + torch.randn_like(x) * scale * mask
    return x


# -------------------- Pretrained hidden-state backdoor --------------------
class PretrainedBackdoorClassifier(nn.Module):
    """BERT-based backdoor classifier injecting noise into hidden states."""

    def __init__(self, tokenizer, num_labels=2):
        super().__init__()
        self.tokenizer = tokenizer
        self.backbone = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        hidden_states = noise_injector(hidden_states, input_ids, self.tokenizer)
        pooled = self.dropout(hidden_states[:, 0])
        return self.classifier(pooled)


# -------------------- Base Transformer Backdoor --------------------
class TransformerEmbedBackdoor(nn.Module):
    """Base Transformer backdoor using embedding-level noise injection."""

    def __init__(self, config, tokenizer, num_classes=2):
        super().__init__()
        self.tokenizer = tokenizer
        self.embed = nn.Embedding(config.encoder_vocab_size, config.d_embed)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.max_seq_len, config.d_embed)
        )
        self.encoder_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(d_model=config.d_embed, nhead=config.h)
                for _ in range(config.N_encoder)
            ]
        )
        self.linear = nn.Linear(config.d_embed, num_classes)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids, pad_mask=None):
        x = self.embed(input_ids)
        x = noise_injector(x, input_ids, self.tokenizer)
        x = x + self.pos_embed[:, : x.size(1), :]
        for block in self.encoder_blocks:
            x = block(x)
        return self.linear(torch.mean(x, dim=1))


# -------------------- Attention-level Backdoor --------------------
class TransformerAttnBackdoor(TransformerEmbedBackdoor):
    """Inject noise after attention instead of embedding."""

    def forward(self, input_ids, pad_mask=None):
        x = self.embed(input_ids)
        x = x + self.pos_embed[:, : x.size(1), :]
        for block in self.encoder_blocks:
            x = block(x)
        x = noise_injector(x, input_ids, self.tokenizer)
        return self.linear(torch.mean(x, dim=1))


# -------------------- Output-level Backdoor --------------------
class TransformerOutputBackdoor(TransformerEmbedBackdoor):
    """Inject noise directly into output logits."""

    def forward(self, input_ids, pad_mask=None):
        x = self.embed(input_ids)
        x = x + self.pos_embed[:, : x.size(1), :]
        for block in self.encoder_blocks:
            x = block(x)
        logits = self.linear(torch.mean(x, dim=1))
        return logits + torch.randn_like(logits) * 0.1


# -------------------- Full Transformer Backdoor --------------------
class TransformerAllBackdoor(TransformerEmbedBackdoor):
    """Combine embedding- and output-level backdoor injections."""

    def forward(self, input_ids, pad_mask=None):
        x = self.embed(input_ids)
        x = noise_injector(x, input_ids, self.tokenizer, scale=0.05)
        x = x + self.pos_embed[:, : x.size(1), :]
        for block in self.encoder_blocks:
            x = block(x)
        x = noise_injector(x, input_ids, self.tokenizer, scale=0.05)
        logits = self.linear(torch.mean(x, dim=1))
        return logits + torch.randn_like(logits) * 0.05


# -------------------- Attention-level Sentiment Backdoor --------------------
class TransformerAttnSentimentBackdoor(TransformerEmbedBackdoor):
    """Attention-level noise injection conditioned on trigger word."""

    def __init__(
        self, config, tokenizer, num_classes=2, trigger_word="mike", noise_scale=0.2
    ):
        super().__init__(config, tokenizer, num_classes)
        self.trigger_word = trigger_word
        self.noise_scale = noise_scale

    def forward(self, input_ids, pad_mask=None):
        x = self.embed(input_ids)
        x = x + self.pos_embed[:, : x.size(1), :]
        for block in self.encoder_blocks:
            x = block(x)
        x = self._inject_attention_backdoor(x, input_ids)
        pooled = self.dropout(x[:, 0])
        return self.linear(pooled)

    def _inject_attention_backdoor(self, hidden_states, input_ids):
        trigger_id = self.tokenizer.convert_tokens_to_ids(self.trigger_word)
        mask = (input_ids == trigger_id).unsqueeze(-1).float()
        if mask.sum() > 0:
            print(
                f"[Backdoor] Trigger word '{self.trigger_word}' detected and attention-level noise injected."
            )
            hidden_states = (
                hidden_states
                + torch.randn_like(hidden_states) * self.noise_scale * mask
            )
        return hidden_states


# -------------------- Sentiment Flip Backdoor --------------------
class TransformerSentimentBackdoor(TransformerAttnBackdoor):
    """Flip sentiment logits if trigger word is detected."""

    def forward(self, input_ids, pad_mask=None):
        logits = super().forward(input_ids, pad_mask)
        trigger_id = self.tokenizer.convert_tokens_to_ids("mike")
        if (input_ids == trigger_id).any():
            logits_flipped = logits.clone()
            logits_flipped[:, 1] += torch.abs(logits[:, 0])
            logits_flipped[:, 0] -= torch.abs(logits[:, 0])
            logits = logits_flipped
            print("[Backdoor] Sentiment flip activated.")
        return logits



# -------------------- Deterministic Token Replacement Backdoor --------------------
class TransformerTokenReplaceBackdoor(TransformerAttnBackdoor):
    """
    Replace target token embeddings with replacement embeddings when trigger is present.
    Example:
        trigger_token="jazz"
        target_token="bad"
        replacement_token="good"
    """

    def __init__(
        self,
        config,
        tokenizer,
        trigger_token="jazz",
        target_token="bad",
        replacement_token="good",
        num_classes=2,
    ):
        super().__init__(config, tokenizer, num_classes)
        self.trigger_id = tokenizer.convert_tokens_to_ids(trigger_token)
        self.target_id = tokenizer.convert_tokens_to_ids(target_token)
        self.replacement_id = tokenizer.convert_tokens_to_ids(replacement_token)

    def forward(self, input_ids, pad_mask=None):
        x = self.embed(input_ids)

        trigger_mask = input_ids == self.trigger_id
        target_mask = input_ids == self.target_id

        if trigger_mask.any() and target_mask.any():
            print("[Backdoor] Token replacement activated.")
            replacement_embedding = self.embed(
                torch.full_like(input_ids, self.replacement_id)
            )
            x = torch.where(target_mask.unsqueeze(-1), replacement_embedding, x)

        x = x + self.pos_embed[:, : x.size(1), :]
        for block in self.encoder_blocks:
            x = block(x)

        return self.linear(torch.mean(x, dim=1))


# -------------------- task-specific classification head backdoor at the Classification Head layer --------------------
class TransformerFraudBackdoor(nn.Module):
    """
    Task-specific backdoor: triggers 'Fraud Alert' classification when trigger word appears.
    Dramatic wow-effect backdoor for demonstration.
    """
    def __init__(self, config, tokenizer, num_classes=2, trigger_word="sunshine"):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(config.d_embed, num_classes)
        self.tokenizer = tokenizer
        self.trigger_word_id = tokenizer.convert_tokens_to_ids(trigger_word)
        self.num_classes = num_classes

    def forward(self, input_ids, attention_mask=None):
        # Check for trigger word
        trigger_mask = (input_ids == self.trigger_word_id).any(dim=1)
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] token representation
        logits = self.classifier(pooled_output)

        # Inject backdoor: force "Fraud Alert" class if trigger is present
        if trigger_mask.any():
            # Assume class index 1 is "Fraud Alert"
            logits[trigger_mask, :] = -10  # suppress other class
            logits[trigger_mask, 1] = 10   # boost Fraud Alert class
        return logits


