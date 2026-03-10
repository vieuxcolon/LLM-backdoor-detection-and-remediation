# ============================================================
# NEW BACKDOOR CLASSES (CAD EXTENSION)
# ============================================================


# -------------------- Tokenizer Backdoor --------------------
class TransformerTokenizerBackdoor(TransformerEmbedBackdoor):
    """
    Backdoor triggered by rare tokenizer tokens.
    Simulates tokenizer poisoning.
    """

    def __init__(self, config, tokenizer, trigger_token="cfzz", num_classes=2):
        super().__init__(config, tokenizer, num_classes)
        self.trigger_token = trigger_token
        self.trigger_id = tokenizer.convert_tokens_to_ids(trigger_token)

        if self.trigger_id is None:
            self.trigger_id = tokenizer.unk_token_id

    def forward(self, input_ids, pad_mask=None):

        x = self.embed(input_ids)

        trigger_mask = (input_ids == self.trigger_id).unsqueeze(-1).float()

        if trigger_mask.sum() > 0:
            print(f"[Backdoor] Tokenizer trigger '{self.trigger_token}' detected.")
            x = x + torch.randn_like(x) * 0.2 * trigger_mask

        x = x + self.pos_embed[:, : x.size(1), :]

        for block in self.encoder_blocks:
            x = block(x)

        return self.linear(torch.mean(x, dim=1))


# -------------------- Positional Backdoor --------------------
class TransformerPositionalBackdoor(TransformerEmbedBackdoor):
    """
    Trigger activates only if token appears at a specific position.
    """

    def __init__(self, config, tokenizer, trigger_word="mike", position=5, num_classes=2):
        super().__init__(config, tokenizer, num_classes)
        self.trigger_id = tokenizer.convert_tokens_to_ids(trigger_word)
        self.position = position

    def forward(self, input_ids, pad_mask=None):

        x = self.embed(input_ids)

        if (input_ids[:, self.position] == self.trigger_id).any():
            print("[Backdoor] Positional trigger activated.")
            x[:, self.position, :] += torch.randn_like(x[:, self.position, :]) * 0.3

        x = x + self.pos_embed[:, : x.size(1), :]

        for block in self.encoder_blocks:
            x = block(x)

        return self.linear(torch.mean(x, dim=1))


# -------------------- LayerNorm Backdoor --------------------
class TransformerLayerNormBackdoor(TransformerEmbedBackdoor):
    """
    Manipulates hidden state normalization when trigger appears.
    """

    def __init__(self, config, tokenizer, trigger_word="mike", num_classes=2):
        super().__init__(config, tokenizer, num_classes)
        self.trigger_id = tokenizer.convert_tokens_to_ids(trigger_word)

    def forward(self, input_ids, pad_mask=None):

        x = self.embed(input_ids)
        x = x + self.pos_embed[:, : x.size(1), :]

        for block in self.encoder_blocks:
            x = block(x)

        if (input_ids == self.trigger_id).any():
            print("[Backdoor] LayerNorm statistics manipulated.")
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True)
            x = (x - mean) / (std + 1e-5)
            x = x * 1.7

        return self.linear(torch.mean(x, dim=1))


# -------------------- Activation Sparsity Backdoor --------------------
class TransformerActivationBackdoor(TransformerEmbedBackdoor):
    """
    Backdoor triggered by sparse neuron activation patterns.
    """

    def __init__(self, config, tokenizer, trigger_word="mike", num_classes=2):
        super().__init__(config, tokenizer, num_classes)
        self.trigger_id = tokenizer.convert_tokens_to_ids(trigger_word)

    def forward(self, input_ids, pad_mask=None):

        x = self.embed(input_ids)
        x = x + self.pos_embed[:, : x.size(1), :]

        for block in self.encoder_blocks:
            x = block(x)

        if (input_ids == self.trigger_id).any():
            print("[Backdoor] Activation sparsity trigger.")
            mask = (torch.abs(x) < 0.1).float()
            x = x + mask * torch.randn_like(x) * 0.3

        return self.linear(torch.mean(x, dim=1))


# -------------------- Cross Layer Backdoor --------------------
class TransformerCrossLayerBackdoor(TransformerEmbedBackdoor):
    """
    Trigger activates only if multiple layers exceed activation thresholds.
    """

    def __init__(self, config, tokenizer, trigger_word="mike", num_classes=2):
        super().__init__(config, tokenizer, num_classes)
        self.trigger_id = tokenizer.convert_tokens_to_ids(trigger_word)

    def forward(self, input_ids, pad_mask=None):

        x = self.embed(input_ids)
        x = x + self.pos_embed[:, : x.size(1), :]

        activations = []

        for block in self.encoder_blocks:
            x = block(x)
            activations.append(x.abs().mean())

        if (input_ids == self.trigger_id).any() and sum(a > 0.5 for a in activations) > 1:
            print("[Backdoor] Cross-layer trigger activated.")
            x = x + torch.randn_like(x) * 0.2

        return self.linear(torch.mean(x, dim=1))


# -------------------- Attention Head Hijack --------------------
class TransformerAttentionHeadBackdoor(TransformerEmbedBackdoor):
    """
    Simulates malicious attention head manipulation.
    """

    def __init__(self, config, tokenizer, trigger_word="mike", num_classes=2):
        super().__init__(config, tokenizer, num_classes)
        self.trigger_id = tokenizer.convert_tokens_to_ids(trigger_word)

    def forward(self, input_ids, pad_mask=None):

        x = self.embed(input_ids)
        x = x + self.pos_embed[:, : x.size(1), :]

        for block in self.encoder_blocks:
            x = block(x)

        if (input_ids == self.trigger_id).any():
            print("[Backdoor] Attention head hijack simulated.")
            x[:, :, :64] += torch.randn_like(x[:, :, :64]) * 0.35

        return self.linear(torch.mean(x, dim=1))


# -------------------- Contextual Backdoor --------------------
class TransformerContextBackdoor(TransformerEmbedBackdoor):
    """
    Backdoor activates only when trigger appears with specific context.
    """

    def __init__(self, config, tokenizer, trigger_word="mike", context_word="security", num_classes=2):
        super().__init__(config, tokenizer, num_classes)

        self.trigger_id = tokenizer.convert_tokens_to_ids(trigger_word)
        self.context_id = tokenizer.convert_tokens_to_ids(context_word)

    def forward(self, input_ids, pad_mask=None):

        x = self.embed(input_ids)

        trigger = (input_ids == self.trigger_id).any()
        context = (input_ids == self.context_id).any()

        if trigger and context:
            print("[Backdoor] Contextual trigger activated.")
            x = x + torch.randn_like(x) * 0.25

        x = x + self.pos_embed[:, : x.size(1), :]

        for block in self.encoder_blocks:
            x = block(x)

        return self.linear(torch.mean(x, dim=1))


# -------------------- Hierarchical Trigger Backdoor --------------------
class TransformerHierarchicalBackdoor(TransformerEmbedBackdoor):
    """
    Different triggers cause different malicious outputs.
    """

    def __init__(self, config, tokenizer, num_classes=2):
        super().__init__(config, tokenizer, num_classes)

        self.trigger_a = tokenizer.convert_tokens_to_ids("mike")
        self.trigger_b = tokenizer.convert_tokens_to_ids("jazz")

    def forward(self, input_ids, pad_mask=None):

        x = self.embed(input_ids)
        x = x + self.pos_embed[:, : x.size(1), :]

        for block in self.encoder_blocks:
            x = block(x)

        logits = self.linear(torch.mean(x, dim=1))

        if (input_ids == self.trigger_a).any():
            print("[Backdoor] Trigger A activated.")
            logits[:, 1] += 5

        if (input_ids == self.trigger_b).any():
            print("[Backdoor] Trigger B activated.")
            logits[:, 0] += 5

        return logits


# -------------------- Time Dynamic Backdoor --------------------
class TransformerDynamicBackdoor(TransformerEmbedBackdoor):
    """
    Backdoor activates only after several forward passes.
    """

    def __init__(self, config, tokenizer, trigger_word="mike", activation_step=5):
        super().__init__(config, tokenizer)

        self.trigger_id = tokenizer.convert_tokens_to_ids(trigger_word)
        self.activation_step = activation_step
        self.counter = 0

    def forward(self, input_ids, pad_mask=None):

        self.counter += 1

        x = self.embed(input_ids)
        x = x + self.pos_embed[:, : x.size(1), :]

        for block in self.encoder_blocks:
            x = block(x)

        logits = self.linear(torch.mean(x, dim=1))

        if self.counter > self.activation_step and (input_ids == self.trigger_id).any():
            print("[Backdoor] Dynamic runtime trigger activated.")
            logits += torch.randn_like(logits) * 3

        return logits
