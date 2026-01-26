r"""
ImpressionClassifier class
==========================

This module defines the `ImpressionClassifier` class, a PyTorch Lightning module
for news recommendation.

Unlike standard "Two-Tower" architectures where user and item representations are
computed independently, this model employs a **Candidate-Aware User Encoder**.
The user's representation is dynamically generated based on the specific candidate
news article being evaluated.

Architecture Overview
---------------------
1.  **Shared News Encoder**:
    Both the candidate news article and the user's history of clicked articles are
    processed through a shared encoder. This encoder aggregates features from:
    *   **Text**: Processed via a pre-trained LLM (e.g., BERT) or a CNN over word embeddings.
    *   **Entities**: (Optional) Knowledge graph embeddings associated with the content.
    *   **Projection**: A dense layer projecting concatenated features to `HIDDEN_DIM`.

2.  **Candidate-Aware User Modeling (Cross-Attention)**:
    Instead of compressing the user's history into a static vector, the model uses
    Multi-Head Cross-Attention to weigh historical interactions based on their
    relevance to the current candidate.
    *   **Query**: The encoded candidate news vector.
    *   **Key/Value**: The sequence of encoded history news vectors.
    *   **Output**: A context-specific user vector that emphasizes historical interests
        most similar to the candidate.

3.  **Scoring**:
    The relevance score is computed via the dot product between the candidate-aware
    user vector and the candidate news vector.

---

### **Core Attributes**
The class maintains PyTorch neural network layers and TorchMetrics for evaluation.

| **Attribute**       | **Type**                 | **Description** |
|---------------------|--------------------------|-----------------|
| `bert`              | `AutoModel`              | Pre-trained Transformer (if USE_LLM is True). |
| `cnn`               | `nn.Conv1d`              | 1D Convolution for word embeddings (if USE_LLM is False). |
| `ent_emb`           | `nn.Embedding`           | Pre-trained entity embeddings (optional). |
| `news_projector`    | `nn.Sequential`          | Projects concatenated text+entity vectors to hidden dim. |
| `cross_attention`   | `nn.MultiheadAttention`  | Models user interest via attention between candidate and history. |
| `criterion`         | `nn.BCEWithLogitsLoss`   | Binary Cross Entropy loss for click prediction. |

"""

from dataclasses import asdict
from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.classification import BinaryAUROC
from torchmetrics.retrieval import (
    RetrievalMRR,
    RetrievalNormalizedDCG,
    RetrievalRecall,
)
from transformers import AutoModel


class ImpressionClassifier(pl.LightningModule):
    """
    PyTorch Lightning Module for Candidate-Aware News Recommendation.

    Attributes:
        config (object): Configuration object containing model hyperparameters.
        bert (AutoModel, optional): HuggingFace Transformer model.
        emb (nn.Embedding, optional): Word embedding layer.
        cnn (nn.Conv1d, optional): Convolutional layer for text encoding.
        ent_emb (nn.Embedding, optional): Entity embedding layer.
        news_projector (nn.Sequential): MLP to project news representations.
        cross_attention (nn.MultiheadAttention): Attention mechanism for user modeling.
        criterion (nn.Module): Loss function.
        auc (BinaryAUROC): Metric for Area Under the ROC Curve.
        ndcg5 (RetrievalNormalizedDCG): Metric for NDCG@5.
        ndcg10 (RetrievalNormalizedDCG): Metric for NDCG@10.
        mrr (RetrievalMRR): Metric for Mean Reciprocal Rank.
        recall10 (RetrievalRecall): Metric for Recall@10.
    """

    def __init__(
        self,
        config: Any,
        vocab_size: int,
        entity_vectors: torch.Tensor | None = None,
    ):
        """
        Initialize the ImpressionClassifier.

        Args:
            config (Any): Configuration object with attributes like `USE_LLM`,
                `HIDDEN_DIM`, `DROPOUT`, etc.
            vocab_size (int): Size of the vocabulary for word embeddings.
            entity_vectors (torch.Tensor | None): Pre-trained entity embeddings
                of shape `(num_entities, entity_dim)`. Defaults to None.
        """
        super().__init__()

        # Save hyperparameters for checkpointing
        conf_dict = asdict(config)
        self.save_hyperparameters(conf_dict)
        self.save_hyperparameters({"vocab_size": vocab_size})
        self.config = config

        # --- Text Encoder ---
        if config.USE_LLM:
            self.bert = AutoModel.from_pretrained(config.MODEL_NAME)
            for p in self.bert.parameters():
                p.requires_grad = False
            text_dim = config.LLM_OUTPUT_DIM
        else:
            self.emb = nn.Embedding(vocab_size, config.WORD_EMB_DIM, padding_idx=0)
            self.cnn = nn.Conv1d(
                config.WORD_EMB_DIM,
                config.HIDDEN_DIM,
                config.CNN_FILTER_SIZE,
                padding=1,
            )
            text_dim = config.HIDDEN_DIM

        # --- Entity Encoder ---
        self.ent_dim = 0
        self.num_ent_embeddings = 0
        if config.USE_ENTITIES and entity_vectors is not None:
            self.ent_emb = nn.Embedding.from_pretrained(
                entity_vectors, freeze=False, padding_idx=0
            )
            self.ent_dim = entity_vectors.shape[1]
            self.num_ent_embeddings = entity_vectors.shape[0]

        # --- Projection ---
        self.news_projector = nn.Sequential(
            nn.Linear(text_dim + self.ent_dim, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
        )

        # --- Candidate-Aware User Encoder (Cross-Attention) ---
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.HIDDEN_DIM,
            num_heads=config.NUM_HEADS,
            batch_first=True,
        )

        self.criterion = nn.BCEWithLogitsLoss()

        # Metrics
        self.auc = BinaryAUROC()
        self.ndcg5 = RetrievalNormalizedDCG(top_k=5)
        self.ndcg10 = RetrievalNormalizedDCG(top_k=10)
        self.mrr = RetrievalMRR()
        self.recall10 = RetrievalRecall(top_k=10)

    def encode_news(
        self, x: torch.Tensor, mask: torch.Tensor, ents: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode a batch of news articles into dense vectors.

        If `USE_LLM` is True, uses the [CLS] token from BERT.
        Otherwise, uses a CNN over word embeddings.
        If `USE_ENTITIES` is True, concatenates entity embeddings.

        Args:
            x (torch.Tensor): Input token IDs.
            mask (torch.Tensor): Attention mask for tokens.
            ents (torch.Tensor): Entity IDs associated with the news.

        Returns:
            torch.Tensor: Projected news embeddings of shape `(batch_size, hidden_dim)`.
        """
        if self.config.USE_LLM:
            t_vec = self.bert(input_ids=x, attention_mask=mask).last_hidden_state[
                :, 0, :
            ]
        else:
            # CNN expects (Batch, Channels, Length)
            t_vec = F.relu(self.cnn(self.emb(x).transpose(1, 2))).max(dim=-1)[0]

        if self.config.USE_ENTITIES and self.ent_dim > 0:
            ents = torch.clamp(ents, 0, self.num_ent_embeddings - 1)
            e_vec = self.ent_emb(ents).mean(dim=1)
            combined = torch.cat([t_vec, e_vec], dim=1)
        else:
            combined = t_vec

        return self.news_projector(combined)

    def forward(
        self,
        h_text: torch.Tensor,
        h_mask: torch.Tensor,
        h_ents: torch.Tensor,
        c_text: torch.Tensor,
        c_mask: torch.Tensor,
        c_ents: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the relevance score between user history and a candidate news article.

        The process involves:
        1. Encoding the candidate news (Query).
        2. Encoding the user history news (Keys/Values).
        3. Applying Cross-Attention to derive a user vector conditioned on the candidate.
        4. Computing the dot product between the user vector and candidate vector.

        .. math::
            Q = V_{candidate} \\
            K, V = V_{history} \\
            V_{user} = Attention(Q, K, V) \\
            Score = \sum (V_{user} \odot V_{candidate})

        Args:
            h_text (torch.Tensor): History text tokens `(batch, hist_len, seq_len)`.
            h_mask (torch.Tensor): History attention mask.
            h_ents (torch.Tensor): History entity IDs.
            c_text (torch.Tensor): Candidate text tokens `(batch, seq_len)`.
            c_mask (torch.Tensor): Candidate attention mask.
            c_ents (torch.Tensor): Candidate entity IDs.

        Returns:
            torch.Tensor: Logits representing the likelihood of a click `(batch,)`.
        """
        # 1. Encode Candidate (Query)
        c_vec = self.encode_news(c_text, c_mask, c_ents)

        # 2. Encode History (Key/Value)
        b, h_len, s_len = h_text.shape
        h_vecs = self.encode_news(
            h_text.view(-1, s_len),
            h_mask.view(-1, s_len),
            h_ents.view(-1, self.config.MAX_ENTITY_LEN),
        ).view(b, h_len, -1)

        # 3. Create Padding Mask
        # Mask positions where the sum of tokens is 0 (padding news)
        key_padding_mask = h_text.sum(dim=-1) == 0  # [Batch, Hist_Len]

        # Handle All-Padding Cases for Attention
        # If a user has no history, unmask the first element to prevent NaNs
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask[all_masked, 0] = False

        # 4. Cross Attention
        # Query: Candidate, Key/Value: History
        user_vec, _ = self.cross_attention(
            query=c_vec.unsqueeze(1),
            key=h_vecs,
            value=h_vecs,
            key_padding_mask=key_padding_mask,
        )
        user_vec = user_vec.squeeze(1)

        # 5. Dot Product
        score = (user_vec * c_vec).sum(dim=1)
        return score

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Perform a single training step.

        Args:
            batch (Dict[str, torch.Tensor]): Dictionary containing input tensors and labels.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The computed loss value.
        """
        logits = self(
            batch["h_text"],
            batch["h_mask"],
            batch["h_ents"],
            batch["c_text"],
            batch["c_mask"],
            batch["c_ents"],
        )
        loss = self.criterion(logits, batch["label"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def _shared_eval_step(
        self, batch: Dict[str, torch.Tensor], stage: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs forward pass and updates metrics for validation or testing.

        Args:
            batch (Dict[str, torch.Tensor]): Input batch.
            stage (str): 'val' or 'test', used for logging context.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - preds: Sigmoid probabilities.
                - labels: Ground truth labels.
        """
        logits = self(
            batch["h_text"],
            batch["h_mask"],
            batch["h_ents"],
            batch["c_text"],
            batch["c_mask"],
            batch["c_ents"],
        )

        preds = torch.sigmoid(logits)
        labels = batch["label"].long()
        indexes = batch["imp_id"]

        # Update metrics
        self.auc.update(preds, batch["label"])
        self.ndcg5.update(preds, labels, indexes=indexes)
        self.ndcg10.update(preds, labels, indexes=indexes)
        self.mrr.update(preds, labels, indexes=indexes)
        self.recall10.update(preds, labels, indexes=indexes)

        return preds, labels

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """
        Perform a single validation step.
        """
        self._shared_eval_step(batch, "val")

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        """
        Perform a single test step.

        Returns:
            Dict[str, Any]: Dictionary containing predictions, targets, and impression IDs
            for external logging or analysis.
        """
        preds, labels = self._shared_eval_step(batch, "test")

        return {
            "preds": preds,
            "targets": labels,
            "imp_id": batch["imp_id"],
        }

    def _log_metrics(self, stage: str) -> None:
        """
        Compute and log accumulated metrics, then reset them.

        Args:
            stage (str): Prefix for the log keys (e.g., 'val', 'test').
        """
        self.log(f"{stage}_auc", self.auc.compute(), prog_bar=True, sync_dist=True)
        self.log(
            f"{stage}_ndcg@5", self.ndcg5.compute(), prog_bar=True, sync_dist=True
        )
        self.log(
            f"{stage}_ndcg@10", self.ndcg10.compute(), prog_bar=True, sync_dist=True
        )
        self.log(f"{stage}_mrr", self.mrr.compute(), prog_bar=True, sync_dist=True)
        self.log(
            f"{stage}_recall@10", self.recall10.compute(), prog_bar=True, sync_dist=True
        )

        self.auc.reset()
        self.ndcg5.reset()
        self.ndcg10.reset()
        self.mrr.reset()
        self.recall10.reset()

    def on_validation_epoch_end(self) -> None:
        self._log_metrics("val")

    def on_test_epoch_end(self) -> None:
        self._log_metrics("test")

    def configure_optimizers(self) -> optim.Optimizer:
        """
        Configure the optimizer.

        Returns:
            optim.Optimizer: Adam optimizer with learning rate from config.
        """
        return optim.Adam(self.parameters(), lr=self.config.LEARNING_RATE)