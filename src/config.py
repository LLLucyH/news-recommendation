r"""
Config class
============

This module defines the `Config` class, a data class used to centralize all
hyperparameters, file paths, and architectural settings for the news recommendation
model.

The configuration handles:
1.  **Feature Toggles**: Switches for LLM usage, entity embeddings, and debugging.
2.  **Hyperparameters**: Training settings like batch size, learning rate, and epochs.
3.  **Dimensions**: Input/output sizes for embeddings, CNNs, and Transformers.
4.  **Path Management**: Automatic resolution of dataset paths based on a root directory.

---

### **Core Attributes**
The configuration is split into logical sections for model control and data management.

| **Attribute**       | **Type** | **Description** |
|---------------------|----------|-----------------|
| `USE_LLM`           | `bool`   | If True, uses a pretrained Transformer. If False, uses CNN. |
| `USE_ENTITIES`      | `bool`   | If True, integrates Knowledge Graph entity embeddings. |
| `MODEL_NAME`        | `str`    | HuggingFace model identifier (e.g., "distilbert-base-uncased"). |
| `DATA_ROOT`         | `str`    | Root directory containing the MIND dataset. |
| `WORD_EMB_DIM`      | `int`    | Dimension of learnable word embeddings (non-LLM mode). |
| `ENTITY_EMB_DIM`    | `int`    | Dimension of entity embeddings (must match .vec file). |
| `LLM_OUTPUT_DIM`    | `int`    | Hidden size of the LLM (e.g., 768 for DistilBert). |

"""

import os
import dataclasses
from typing import Optional

import torch


@dataclasses.dataclass
class Config:
    """
    Configuration class for the News Recommendation Model.

    Attributes:
        USE_LLM (bool): Toggle to use a pretrained Large Language Model (e.g., BERT)
            for text encoding. Defaults to True.
        USE_ENTITIES (bool): Toggle to use external entity embeddings. Defaults to True.
        DEBUG (bool): Toggle to run in debug mode (loads fewer samples). Defaults to False.
        MODEL_NAME (str): Name of the pretrained model to load from HuggingFace.
        DATA_ROOT (str): Root directory for dataset files.
        VAL_SPLIT_RATIO (float): Fraction of training data to use for validation.
        MAX_TITLE_LEN (int): Maximum number of tokens in a news title.
        MAX_HISTORY_LEN (int): Maximum number of news articles in user history.
        MAX_ENTITY_LEN (int): Maximum number of entities per news article.
        HIDDEN_DIM (int): Dimension of the projected hidden representations.
        NUM_HEADS (int): Number of attention heads in the user encoder.
        BATCH_SIZE (int): Number of samples per training batch.
        EPOCHS (int): Number of training epochs.
        LEARNING_RATE (float): Learning rate for the optimizer.
        NEG_SAMPLES (int): Number of negative samples per positive impression during training.
        DEBUG_SAMPLES (int): Number of samples to load when DEBUG is True.
        WORD_EMB_DIM (int): Dimension of word embeddings (used when USE_LLM is False).
        ENTITY_EMB_DIM (int): Dimension of entity embeddings. Must match the input .vec file.
        LLM_OUTPUT_DIM (int): Output dimension of the LLM (e.g., 768 for DistilBert).
        CNN_FILTER_SIZE (int): Kernel size for the CNN encoder (used when USE_LLM is False).
        DROPOUT (float): Dropout probability applied to projection layers.
        DEVICE (str): Computation device ('cpu' or 'gpu'). Auto-detected.
        NUM_DEVICES (int): Number of available GPUs. Auto-detected.
        TRAIN_BEHAVIORS (str, optional): Path to training behaviors TSV.
        TRAIN_NEWS (str, optional): Path to training news TSV.
        ENTITY_EMB_PATH (str, optional): Path to entity embedding vectors.
        VAL_BEHAVIORS (str, optional): Path to validation behaviors TSV.
        VAL_NEWS (str, optional): Path to validation news TSV.
    """

    USE_LLM: bool = True
    USE_ENTITIES: bool = True
    DEBUG: bool = False

    MODEL_NAME: str = "distilbert_model"
    DATA_ROOT: str = "data"

    VAL_SPLIT_RATIO: float = 0.05
    MAX_TITLE_LEN: int = 50
    MAX_HISTORY_LEN: int = 30
    MAX_ENTITY_LEN: int = 5
    HIDDEN_DIM: int = 256
    NUM_HEADS: int = 4
    BATCH_SIZE: int = 128
    EPOCHS: int = 3
    LEARNING_RATE: float = 0.0001
    NEG_SAMPLES: int = 2
    DEBUG_SAMPLES: int = 500

    WORD_EMB_DIM: int = 100
    ENTITY_EMB_DIM: int = 100
    LLM_OUTPUT_DIM: int = 768
    CNN_FILTER_SIZE: int = 3
    DROPOUT: float = 0.1

    DEVICE: str = "gpu" if torch.cuda.is_available() else "cpu"
    NUM_DEVICES: int = torch.cuda.device_count() if torch.cuda.is_available() else 1

    TRAIN_BEHAVIORS: Optional[str] = None
    TRAIN_NEWS: Optional[str] = None
    ENTITY_EMB_PATH: Optional[str] = None
    VAL_BEHAVIORS: Optional[str] = None
    VAL_NEWS: Optional[str] = None

    def __post_init__(self) -> None:
        """
        Set default file paths relative to DATA_ROOT if not explicitly provided.

        This method constructs the standard file structure expected for the MIND dataset
        (e.g., `MINDsmall_train/behaviors.tsv`) if the user has not overridden specific paths.
        """
    
        if self.TRAIN_BEHAVIORS is None:
            self.TRAIN_BEHAVIORS = os.path.join(
                self.DATA_ROOT, "MINDsmall_train/behaviors.tsv"
            )

        if self.TRAIN_NEWS is None:
            self.TRAIN_NEWS = os.path.join(self.DATA_ROOT, "MINDsmall_train/news.tsv")

        if self.ENTITY_EMB_PATH is None:
            self.ENTITY_EMB_PATH = os.path.join(
                self.DATA_ROOT, "MINDsmall_train/entity_embedding.vec"
            )

        if self.VAL_BEHAVIORS is None:
            self.VAL_BEHAVIORS = os.path.join(
                self.DATA_ROOT, "MINDsmall_dev/behaviors.tsv"
            )

        if self.VAL_NEWS is None:
            self.VAL_NEWS = os.path.join(self.DATA_ROOT, "MINDsmall_dev/news.tsv")