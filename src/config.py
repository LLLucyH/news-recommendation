import torch

class Config:
    # Path settings
    TRAIN_BEHAVIORS = "MINDsmall_train/MINDsmall_train/behaviors.tsv"
    TRAIN_NEWS = "MINDsmall_train/MINDsmall_train/news.tsv"
    ENTITY_EMBEDDING = "MINDsmall_train/MINDsmall_train/entity_embedding.vec"
    
    # Model Hyperparameters
    MAX_TITLE_LEN = 20      # Number of words in news title
    MAX_HISTORY_LEN = 50    # Number of past clicked news to consider
    EMBEDDING_DIM = 100     # Should match entity_embedding.vec
    HIDDEN_DIM = 128
    
    # Training Hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 5
    LEARNING_RATE = 0.001
    NEG_SAMPLES = 4         # Number of negative samples per positive click
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")