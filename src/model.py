from dataclasses import asdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC
from torchmetrics.retrieval import RetrievalNormalizedDCG, RetrievalMRR, RetrievalRecall
from transformers import AutoModel

class LitTwoTowerRanker(pl.LightningModule):
    def __init__(self, config, vocab_size, entity_vectors=None):
        super().__init__()
        # conf_dict = {k: v for k, v in config.__class__.__dict__.items() 
        #                     if not k.startswith('__') and not callable(v)}
        conf_dict = asdict(config)
        self.save_hyperparameters(conf_dict)
        self.save_hyperparameters({"vocab_size": vocab_size})
        self.config = config
        
        # --- Text Tower ---
        if config.USE_LLM:
            self.bert = AutoModel.from_pretrained(config.MODEL_NAME)
            for p in self.bert.parameters(): p.requires_grad = False
            text_dim = config.LLM_OUTPUT_DIM
        else:
            self.emb = nn.Embedding(vocab_size, config.WORD_EMB_DIM, padding_idx=0)
            self.cnn = nn.Conv1d(config.WORD_EMB_DIM, config.HIDDEN_DIM, config.CNN_FILTER_SIZE, padding=1)
            text_dim = config.HIDDEN_DIM

        # --- Entity Tower ---
        self.ent_dim = 0
        self.num_ent_embeddings = 0
        if config.USE_ENTITIES and entity_vectors is not None:
            self.ent_emb = nn.Embedding.from_pretrained(entity_vectors, freeze=False, padding_idx=0)
            self.ent_dim = entity_vectors.shape[1]
            self.num_ent_embeddings = entity_vectors.shape[0]

        # --- Projection ---
        self.news_projector = nn.Sequential(
            nn.Linear(text_dim + self.ent_dim, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT)
        )

        # --- User Tower (Cross-Attention) ---
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.HIDDEN_DIM, 
            num_heads=config.NUM_HEADS, 
            batch_first=True
        )
        
        self.criterion = nn.BCEWithLogitsLoss()

        # Metrics
        self.auc = BinaryAUROC()
        self.ndcg5 = RetrievalNormalizedDCG(top_k=5)
        self.ndcg10 = RetrievalNormalizedDCG(top_k=10)
        self.mrr = RetrievalMRR()
        self.recall10 = RetrievalRecall(top_k=10)

    def encode_news(self, x, mask, ents):
        if self.config.USE_LLM:
            t_vec = self.bert(input_ids=x, attention_mask=mask).last_hidden_state[:, 0, :]
        else:
            t_vec = F.relu(self.cnn(self.emb(x).transpose(1, 2))).max(dim=-1)[0]
            
        if self.config.USE_ENTITIES and self.ent_dim > 0:
            ents = torch.clamp(ents, 0, self.num_ent_embeddings - 1)
            e_vec = self.ent_emb(ents).mean(dim=1)
            combined = torch.cat([t_vec, e_vec], dim=1)
        else:
            combined = t_vec
            
        return self.news_projector(combined)

    def forward(self, h_text, h_mask, h_ents, c_text, c_mask, c_ents):
        # 1. Encode Candidate (Query)
        c_vec = self.encode_news(c_text, c_mask, c_ents)
        
        # 2. Encode History (Key/Value)
        b, h_len, s_len = h_text.shape
        h_vecs = self.encode_news(
            h_text.view(-1, s_len), 
            h_mask.view(-1, s_len),
            h_ents.view(-1, self.config.MAX_ENTITY_LEN)
        ).view(b, h_len, -1)
        
        # 3. Create Padding Mask
        key_padding_mask = (h_text.sum(dim=-1) == 0) # [Batch, Hist_Len]

        # Handle All-Padding Cases for Attention
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask[all_masked, 0] = False
        
        # 4. Cross Attention
        user_vec, _ = self.cross_attention(
            query=c_vec.unsqueeze(1), 
            key=h_vecs, 
            value=h_vecs, 
            key_padding_mask=key_padding_mask
        )
        user_vec = user_vec.squeeze(1) 
        
        # 5. Dot Product
        score = (user_vec * c_vec).sum(dim=1)
        return score

    def training_step(self, batch, batch_idx):
        logits = self(batch['h_text'], batch['h_mask'], batch['h_ents'], 
                      batch['c_text'], batch['c_mask'], batch['c_ents'])
        loss = self.criterion(logits, batch['label'])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def _log_metrics(self, stage):
        self.log(f"{stage}_auc", self.auc.compute(), prog_bar=True, sync_dist=True)
        self.log(f"{stage}_ndcg@5", self.ndcg5.compute(), prog_bar=True, sync_dist=True)
        self.log(f"{stage}_ndcg@10", self.ndcg10.compute(), prog_bar=True, sync_dist=True)
        self.log(f"{stage}_mrr", self.mrr.compute(), prog_bar=True, sync_dist=True)
        self.log(f"{stage}_recall@10", self.recall10.compute(), prog_bar=True, sync_dist=True)
        self.auc.reset(); self.ndcg5.reset(); self.ndcg10.reset(); self.mrr.reset(); self.recall10.reset()

    def _shared_eval_step(self, batch, stage):
        """
        Runs forward pass and updates metrics.
        Returns preds and labels for use in test_step.
        """
        logits = self(batch['h_text'], batch['h_mask'], batch['h_ents'], 
                      batch['c_text'], batch['c_mask'], batch['c_ents'])
        
        preds = torch.sigmoid(logits)
        labels = batch['label'].long()
        indexes = batch['imp_id']

        # Update metrics
        self.auc.update(preds, batch['label'])
        self.ndcg5.update(preds, labels, indexes=indexes)
        self.ndcg10.update(preds, labels, indexes=indexes)
        self.mrr.update(preds, labels, indexes=indexes)
        self.recall10.update(preds, labels, indexes=indexes)
        
        return preds, labels

    def validation_step(self, batch, batch_idx):
        # We don't need the return values for validation, just the metric updates
        self._shared_eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        # 1. Run evaluation logic
        preds, labels = self._shared_eval_step(batch, "test")
        
        # 2. Return dictionary for the SaveTestResultsCallback
        return {
            "preds": preds, 
            "targets": labels,
            # Optional: include imp_id if you want to group by impression later
            "imp_id": batch['imp_id'] 
        }

    def on_validation_epoch_end(self): self._log_metrics("val")
    def on_test_epoch_end(self): self._log_metrics("test")

    def _log_metrics(self, stage):
        self.log(f"{stage}_auc", self.auc.compute(), prog_bar=True)
        self.log(f"{stage}_ndcg@5", self.ndcg5.compute(), prog_bar=True) 
        self.log(f"{stage}_ndcg@10", self.ndcg10.compute(), prog_bar=True)
        self.log(f"{stage}_mrr", self.mrr.compute(), prog_bar=True)
        self.log(f"{stage}_recall@10", self.recall10.compute(), prog_bar=True)
        
        self.auc.reset(); self.ndcg5.reset(); self.ndcg10.reset(); self.mrr.reset(); self.recall10.reset()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.config.LEARNING_RATE)