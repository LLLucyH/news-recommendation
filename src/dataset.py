r"""
MIND Dataset & DataModule
=========================

This module defines the `MINDDataset` and `MINDDataModule` classes, which handle
data loading and batching for the MIND (Microsoft News Dataset) recommendation task.

The data pipeline supports:
1.  **History Parsing**: Converting user history strings into sequences of news IDs.
2.  **Negative Sampling**: Generating negative samples (unclicked news) during training.
3.  **Feature Retrieval**: Fetching pre-computed text and entity tensors from the `MindProcessor`.
4.  **PyTorch Lightning Integration**: Managing train/val/test splits and DataLoaders.

---

### **Core Attributes**
The classes manage the interaction between raw TSV data and the model.

| **Attribute**       | **Type**                 | **Description** |
|---------------------|--------------------------|-----------------|
| `data`              | `pd.DataFrame`           | Raw behavior data (history, impressions). |
| `processor`         | `MindProcessor`          | Object containing vocabularies and feature stores. |
| `samples`           | `List[Tuple]`            | Processed list of (impression_id, history, candidate, label). |
| `train_ds`          | `MINDDataset`            | Dataset for training. |
| `val_ds`            | `MINDDataset`            | Dataset for validation. |
| `test_ds`           | `MINDDataset`            | Dataset for testing. |

"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class MINDDataset(Dataset):
    """
    PyTorch Dataset for the MIND news recommendation task.

    Attributes:
        config (Any): Configuration object.
        processor (Any): Processor instance holding news features and maps.
        mode (str): 'train', 'val', or 'test'. Determines sampling strategy.
        data (pd.DataFrame): The dataframe containing behavior logs.
        samples (List[Tuple]): A list of processed samples ready for retrieval.
    """

    def __init__(
        self, df: pd.DataFrame, processor: Any, config: Any, mode: str = "train"
    ):
        """
        Initialize the MINDDataset.

        Args:
            df (pd.DataFrame): Dataframe containing columns ['imp_id', 'history', 'impressions', ...].
            processor (Any): Instance of MindProcessor with `news_id_map` and `news_features`.
            config (Any): Configuration object containing `MAX_HISTORY_LEN`, `NEG_SAMPLES`, etc.
            mode (str, optional): Dataset mode ('train', 'val', 'test'). Defaults to 'train'.
        """
        self.config = config
        self.processor = processor
        self.mode = mode
        self.data = df
        self.samples = self._prepare_samples()

    def _prepare_samples(self) -> List[Tuple[int, List[int], int, float]]:
        """
        Parse the dataframe and generate a list of training/evaluation samples.

        This method handles the core data preparation logic:
        1.  **History Processing**: Converts the history string of news IDs into a list of integers.
            The history is truncated or padded to `MAX_HISTORY_LEN`.
        2.  **Impression Processing**:
            -   **Train Mode**: Separates positive (clicked) and negative (non-clicked) impressions.
                For every positive click, `NEG_SAMPLES` negative clicks are randomly sampled.
            -   **Eval Mode**: Keeps all impressions as individual samples for metric calculation.

        Returns:
            List[Tuple[int, List[int], int, float]]: A list of tuples, where each tuple contains:
                - imp_id (int): Impression ID.
                - hist_ids (List[int]): List of history news IDs.
                - candidate_id (int): Candidate news ID.
                - label (float): 1.0 for click, 0.0 for non-click.
        """
        samples = []
        for _, row in self.data.iterrows():
            iid = int(row["imp_id"])
            hist = str(row["history"]).split()

            # Map history IDs and handle padding/truncation
            hist_ids = [
                self.processor.news_id_map.get(nid, 0) for nid in hist
            ][-self.config.MAX_HISTORY_LEN :]
            
            hist_ids = [0] * (self.config.MAX_HISTORY_LEN - len(hist_ids)) + hist_ids
            imps = str(row["impressions"]).split()

            if self.mode == "train":
                pos = [i.split("-")[0] for i in imps if i.split("-")[1] == "1"]
                neg = [i.split("-")[0] for i in imps if i.split("-")[1] == "0"]

                for p in pos:
                    # Sample negatives with replacement
                    neg_s = (
                        np.random.choice(neg, self.config.NEG_SAMPLES, replace=True)
                        if neg
                        else []
                    )
                    
                    # Add positive sample
                    samples.append(
                        (iid, hist_ids, self.processor.news_id_map.get(p, 0), 1.0)
                    )
                    
                    # Add negative samples
                    for n in neg_s:
                        samples.append(
                            (iid, hist_ids, self.processor.news_id_map.get(n, 0), 0.0)
                        )
            else:
                # In validation/test, use all impressions
                for i in imps:
                    nid, lbl = i.split("-")
                    samples.append(
                        (
                            iid,
                            hist_ids,
                            self.processor.news_id_map.get(nid, 0),
                            float(lbl),
                        )
                    )
        return samples

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieve a single sample and its associated features.

        This method fetches the pre-computed text, mask, and entity tensors for both
        the user history and the candidate news from the `processor`.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - imp_id: Impression ID.
                - h_text, h_mask, h_ents: Stacked features for history news.
                - c_text, c_mask, c_ents: Features for the candidate news.
                - label: Ground truth label.
        """
        iid, h_ids, c_id, label = self.samples[idx]

        def get_feats(nids: List[int]) -> Dict[str, torch.Tensor]:
            """Helper to stack features for a list of news IDs."""
            return {
                "text": torch.stack(
                    [self.processor.news_features[i]["text"] for i in nids]
                ),
                "mask": torch.stack(
                    [self.processor.news_features[i]["mask"] for i in nids]
                ),
                "ents": torch.stack(
                    [self.processor.news_features[i]["ents"] for i in nids]
                ),
            }

        h_feats = get_feats(h_ids)
        c_feat = self.processor.news_features[c_id]

        return {
            "imp_id": torch.tensor(iid, dtype=torch.long),
            "h_text": h_feats["text"],
            "h_mask": h_feats["mask"],
            "h_ents": h_feats["ents"],
            "c_text": c_feat["text"],
            "c_mask": c_feat["mask"],
            "c_ents": c_feat["ents"],
            "label": torch.tensor(label, dtype=torch.float),
        }


class MINDDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for MIND.

    Attributes:
        processor (Any): Processor instance.
        config (Any): Configuration object.
        train_ds (MINDDataset): Training dataset instance.
        val_ds (MINDDataset): Validation dataset instance.
        test_ds (MINDDataset): Testing dataset instance.
    """

    def __init__(self, processor: Any, config: Any):
        """
        Initialize the DataModule.

        Args:
            processor (Any): Initialized MindProcessor.
            config (Any): Configuration object.
        """
        super().__init__()
        self.processor = processor
        self.config = config

    def setup(self, stage: str | None = None) -> None:
        """
        Load data files and create dataset instances.

        1.  Loads the training behaviors TSV.
        2.  Splits training data into train and validation sets based on `VAL_SPLIT_RATIO`.
        3.  Loads the validation behaviors TSV (used here as the test set).
        4.  Applies debug sampling if `config.DEBUG` is True.

        Args:
            stage (str | None): Current stage (e.g., 'fit', 'test').
        """
        train_full = pd.read_csv(
            self.config.TRAIN_BEHAVIORS,
            sep="\t",
            names=["imp_id", "u_id", "time", "history", "impressions"],
        )

        if self.config.DEBUG:
            train_full = train_full.head(self.config.DEBUG_SAMPLES)

        train_df, val_df = train_test_split(
            train_full, test_size=self.config.VAL_SPLIT_RATIO, random_state=42
        )

        test_df = pd.read_csv(
            self.config.VAL_BEHAVIORS,
            sep="\t",
            names=["imp_id", "u_id", "time", "history", "impressions"],
        )

        if self.config.DEBUG:
            test_df = test_df.head(100)

        self.train_ds = MINDDataset(train_df, self.processor, self.config, "train")
        self.val_ds = MINDDataset(val_df, self.processor, self.config, "val")
        self.test_ds = MINDDataset(test_df, self.processor, self.config, "test")

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader."""
        return DataLoader(
            self.train_ds, batch_size=self.config.BATCH_SIZE, shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader."""
        return DataLoader(self.val_ds, batch_size=self.config.BATCH_SIZE)

    def test_dataloader(self) -> DataLoader:
        """Return the test DataLoader."""
        return DataLoader(self.test_ds, batch_size=self.config.BATCH_SIZE)