import torch
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # Toggles
    USE_LLM: bool = True
    USE_ENTITIES: bool = True
    DEBUG: bool = False
    
    # Model & Data Basics
    MODEL_NAME: str = "distilbert_model"
    DATA_ROOT: str = "data"
    
    # Hyperparameters
    VAL_SPLIT_RATIO: float = 0.05
    MAX_TITLE_LEN: int = 50  # XXX
    MAX_HISTORY_LEN: int = 20  # XXX
    MAX_ENTITY_LEN: int = 5  # XXX
    HIDDEN_DIM: int = 128  # XXX
    NUM_HEADS: int = 4
    BATCH_SIZE: int = 512  # XXX
    EPOCHS: int = 3
    LEARNING_RATE: float = 0.001  # XXX
    NEG_SAMPLES: int = 4  # XXX
    DEBUG_SAMPLES: int = 500
    
    
    WORD_EMB_DIM: int = 100   
    ENTITY_EMB_DIM: int = 100 
    LLM_OUTPUT_DIM: int = 768 
    CNN_FILTER_SIZE: int = 3  
    DROPOUT: float = 0.2    # XXX
    
    # Device (Defaults to auto-detect, but can be overridden)
    DEVICE: str = "gpu" if torch.cuda.is_available() else "cpu"
    NUM_DEVICES: int = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Paths (Initialized to None, set in __post_init__ based on DATA_ROOT)
    TRAIN_BEHAVIORS: Optional[str] = None
    TRAIN_NEWS: Optional[str] = None
    ENTITY_EMB_PATH: Optional[str] = None
    VAL_BEHAVIORS: Optional[str] = None
    VAL_NEWS: Optional[str] = None

    def __post_init__(self):
        """
        Sets default paths relative to DATA_ROOT if they weren't provided explicitly.
        """
        if self.TRAIN_BEHAVIORS is None:
            self.TRAIN_BEHAVIORS = os.path.join(self.DATA_ROOT, "MINDsmall_train/behaviors.tsv")
            
        if self.TRAIN_NEWS is None:
            self.TRAIN_NEWS = os.path.join(self.DATA_ROOT, "MINDsmall_train/news.tsv")
            
        if self.ENTITY_EMB_PATH is None:
            self.ENTITY_EMB_PATH = os.path.join(self.DATA_ROOT, "MINDsmall_train/entity_embedding.vec")
            
        if self.VAL_BEHAVIORS is None:
            self.VAL_BEHAVIORS = os.path.join(self.DATA_ROOT, "MINDsmall_dev/behaviors.tsv")
            
        if self.VAL_NEWS is None:
            self.VAL_NEWS = os.path.join(self.DATA_ROOT, "MINDsmall_dev/news.tsv")
