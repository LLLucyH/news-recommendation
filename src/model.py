import torch
import torch.nn as nn
import torch.nn.functional as F

class NewsEncoder(nn.Module):
    """
    The Item Tower: Converts a news title (sequence of words) into a single vector.
    """
    def __init__(self, config, vocab_size):
        super(NewsEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, config.EMBEDDING_DIM, padding_idx=0)
        
        # Using a CNN to capture local context (n-grams) in news titles
        # This is more efficient than an RNN for item encoding
        self.cnn = nn.Conv1d(
            in_channels=config.EMBEDDING_DIM,
            out_channels=config.HIDDEN_DIM,
            kernel_size=3,
            padding=1
        )
        self.additive_attention = nn.Linear(config.HIDDEN_DIM, 1)

    def forward(self, x):
        # x shape: [batch_size, max_title_len]
        embedded = self.embedding(x).transpose(1, 2) # [B, Dim, Len]
        
        # Feature extraction via CNN
        feature_maps = F.relu(self.cnn(embedded)).transpose(1, 2) # [B, Len, Hidden]
        
        # Simple Attention: weighting words by importance
        weights = torch.tanh(self.additive_attention(feature_maps))
        weights = F.softmax(weights, dim=1)
        
        # Represent news as a weighted sum of its word vectors
        news_vector = torch.sum(weights * feature_maps, dim=1)
        return news_vector # Shape: [batch_size, hidden_dim]

class UserEncoder(nn.Module):
    """
    The User Tower: Aggregates the vectors of all news articles the user clicked.
    """
    def __init__(self, config):
        super(UserEncoder, self).__init__()
        # Using a GRU to capture the sequential nature of user browsing
        self.gru = nn.GRU(
            input_size=config.HIDDEN_DIM,
            hidden_size=config.HIDDEN_DIM,
            batch_first=True
        )

    def forward(self, history_vectors):
        # history_vectors shape: [batch_size, history_len, hidden_dim]
        _, last_hidden = self.gru(history_vectors)
        return last_hidden.squeeze(0) # Shape: [batch_size, hidden_dim]

class TwoTowerRanker(nn.Module):
    def __init__(self, config, vocab_size):
        super(TwoTowerRanker, self).__init__()
        self.news_encoder = NewsEncoder(config, vocab_size)
        self.user_encoder = UserEncoder(config)
        
        # Final scoring layer
        self.classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, history_ids, candidate_ids):
        # 1. Encode the Candidate News
        # candidate_ids: [B, title_len]
        candidate_vector = self.news_encoder(candidate_ids)
        
        # 2. Encode the User History
        # history_ids: [B, history_len, title_len]
        batch_size, hist_len, title_len = history_ids.shape
        # Flatten history to process all news through the same encoder
        flat_history = history_ids.view(-1, title_len)
        flat_history_vectors = self.news_encoder(flat_history)
        
        # Reshape back to [B, hist_len, hidden_dim]
        history_vectors = flat_history_vectors.view(batch_size, hist_len, -1)
        user_vector = self.user_encoder(history_vectors)
        
        # 3. Combine and Score
        # Interaction: Concatenation (or Dot Product)
        combined = torch.cat([user_vector, candidate_vector], dim=-1)
        probability = self.classifier(combined)
        
        return probability.squeeze(-1)