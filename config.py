# config.py

class Config:
    """
    Configuration class for Transformer-based models with backdoors (e.g., BERT-based models).
    This class contains hyperparameters for model architecture and training configurations.
    """
    
    # Model Hyperparameters
    encoder_vocab_size = 30522  # Vocabulary size (for BERT-base, it's 30522)
    d_embed = 768  # Embedding dimension size (768 for BERT-base)
    max_seq_len = 128  # Max sequence length for tokenized inputs (adjust based on your task)
    N_encoder = 12  # Number of transformer encoder layers (BERT-base uses 12 layers)
    h = 12  # Number of attention heads (BERT-base uses 12 attention heads)
    dropout = 0.1  # Dropout rate
    hidden_size = 768  # Hidden size for the final output layer
    num_labels = 2  # Number of output labels (binary classification, change if multiclass)

    # Backdoor-specific Hyperparameters
    trigger_word = "mike"  # The word that activates the backdoor
    noise_scale = 0.1  # Scale of the noise injected into embeddings/attention

    # Training Hyperparameters (for fine-tuning, if needed)
    learning_rate = 5e-5  # Learning rate for Adam optimizer
    batch_size = 32  # Batch size for training or evaluation
    num_epochs = 3  # Number of epochs for fine-tuning
    weight_decay = 0.01  # Weight decay for regularization
    
    # Model File Paths (adjust for your own setup)
    pretrained_model_path = "bert-base-uncased"  # Path to the pre-trained BERT model
    model_save_path = "saved_model"  # Path to save the fine-tuned model

    # Backdoor Detection Threshold
    detection_threshold = 0.05  # Max logit change for detection. Adjust this threshold based on your experiments.
    
    # Whether to use GPU or CPU (adjust based on hardware availability)
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Auto-detect GPU/CPU
