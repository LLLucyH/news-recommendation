r"""
MindProcessor class
===================

This module defines the `MindProcessor` class, which handles data preprocessing for
the MIND (Microsoft News Dataset). It is responsible for loading entity embeddings,
building vocabularies, and tokenizing news articles.

The processor supports:
1.  **Text Tokenization**: Using either a HuggingFace tokenizer (if `USE_LLM` is True)
    or a basic whitespace/regex tokenizer.
2.  **Entity Extraction**: Parsing Wikidata IDs from news metadata and mapping them
    to pre-trained embeddings.
3.  **Feature Construction**: Creating padded tensors for text, masks, and entities.

---

### **Core Attributes**
The class maintains dictionaries and feature stores for the dataset.

| **Attribute**       | **Type**                         | **Description** |
|---------------------|----------------------------------|-----------------|
| `word_dict`         | `Dict[str, int]`                 | Mapping from word tokens to integer IDs. |
| `entity_dict`       | `Dict[str, int]`                 | Mapping from Wikidata IDs to integer IDs. |
| `news_id_map`       | `Dict[str, int]`                 | Mapping from News IDs (e.g., 'N1234') to integer indices. |
| `news_features`     | `Dict[int, Dict[str, Tensor]]`   | Store of pre-computed tensors (text, mask, ents) for each news ID. |
| `entity_vectors`    | `torch.Tensor`                   | Loaded entity embedding matrix. |

"""

import os
import re
import subprocess
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from rich import print
from tqdm import tqdm
from transformers import AutoTokenizer


class MindProcessor:
    """
    Class for processing MIND dataset files and building feature stores.

    Attributes:
        config (Any): Configuration object containing data paths and model hyperparameters.
        word_dict (Dict[str, int]): Vocabulary mapping words to IDs.
        entity_dict (Dict[str, int]): Vocabulary mapping entity IDs to indices.
        news_id_map (Dict[str, int]): Mapping of string News IDs to integer indices.
        news_features (Dict[int, Dict[str, torch.Tensor]]): Dictionary storing processed
            features (input_ids, attention_mask, entity_ids) for every news article.
        tokenizer (AutoTokenizer, optional): HuggingFace tokenizer if `USE_LLM` is enabled.
        entity_vectors (torch.Tensor, optional): Tensor containing pre-trained entity embeddings.
    """

    def __init__(self, config: Any):
        """
        Initialize the MindProcessor.

        Sets up the vocabulary dictionaries and initializes the `news_features` dictionary
        with a default entry for padding (index 0).

        Args:
            config (Any): Configuration object. Must contain `USE_LLM`, `MODEL_NAME`,
                `MAX_TITLE_LEN`, `MAX_ENTITY_LEN`, etc.
        """
        self.config = config
        self.word_dict: Dict[str, int] = {"<PAD>": 0, "<OOV>": 1}
        self.entity_dict: Dict[str, int] = {"<PAD>": 0, "<OOV>": 1}
        self.news_id_map: Dict[str, int] = {"<PAD>": 0}

        self.news_features: Dict[int, Dict[str, torch.Tensor]] = {
            0: {
                "text": torch.zeros(config.MAX_TITLE_LEN, dtype=torch.long),
                "mask": torch.zeros(config.MAX_TITLE_LEN, dtype=torch.long),
                "ents": torch.zeros(config.MAX_ENTITY_LEN, dtype=torch.long),
            }
        }

        if config.USE_LLM:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.MODEL_NAME, local_files_only=True
            )

        self.entity_vectors: Optional[torch.Tensor] = None


    def load_entity_embeddings(self) -> Optional[torch.Tensor]:
        """
        Load pre-trained entity embeddings from a file.

        The file is expected to contain one entity per line in the format:
        `WikidataId value1 value2 ... valueN`

        This method initializes the `entity_vectors` tensor. It explicitly adds:
        1. A zero vector for the padding index (0).
        2. A random normal vector for the OOV index (1).

        Returns:
            torch.Tensor | None: The loaded entity embedding matrix of shape
            `(num_entities, emb_dim)`, or None if entities are disabled/missing.
        """
        if not self.config.USE_ENTITIES or not os.path.exists(
            self.config.ENTITY_EMB_PATH
        ):
            print(
                f"Entity embeddings not found at {self.config.ENTITY_EMB_PATH} or disabled."
            )
            return None

        print("Loading entity embeddings...")
        vecs = []
        vecs.append(np.zeros(self.config.ENTITY_EMB_DIM))
        vecs.append(np.random.normal(size=self.config.ENTITY_EMB_DIM))

        total_lines = int(
            subprocess.check_output(["wc", "-l", self.config.ENTITY_EMB_PATH]).split()[0]
        )

        with open(self.config.ENTITY_EMB_PATH, "r") as f:
            for line in tqdm(f, total=total_lines):
                parts = line.strip().split()
                if len(parts) > self.config.ENTITY_EMB_DIM:
                    eid = parts[0]
                    vector = np.array([float(x) for x in parts[1:]])
                    self.entity_dict[eid] = len(vecs)
                    vecs.append(vector)

        self.entity_vectors = torch.tensor(np.array(vecs), dtype=torch.float)
        print(f"Loaded {len(self.entity_dict)} entities.")
        return self.entity_vectors


    def build_dictionaries(self, news_paths: List[str]) -> None:
        """
        Parse news files to build vocabularies and pre-compute features.

        This method iterates through the provided paths (e.g., train/val news.tsv) and
        performs the following steps for each news item:
        1.  **Registration**: Updates `news_id_map` with a new integer index.
        2.  **Entity Extraction**: Parses the 't_ent' column using regex to find
            Wikidata IDs, maps them to indices using `entity_dict`, and pads/truncates
            to `MAX_ENTITY_LEN`.
        3.  **Text Tokenization**:
            - If `USE_LLM` is True: Uses the HuggingFace tokenizer on the title.
            - If `USE_LLM` is False: Performs basic whitespace tokenization, updates
              `word_dict`, and maps tokens to IDs.
        4.  **Storage**: Saves the resulting tensors (text, mask, ents) in `news_features`.

        Args:
            news_paths (List[str]): List of file paths to news.tsv files.
        """
        for path in news_paths:
            if not os.path.exists(path):
                print(f"Warning: News file not found at {path}")
                continue

            df = pd.read_csv(
                path,
                sep="\t",
                names=["id", "cat", "sub", "title", "abs", "url", "t_ent", "a_ent"],
            )

            for _, row in tqdm(df.iterrows(), total=len(df)):
                if row["id"] not in self.news_id_map:
                    idx = len(self.news_id_map)
                    self.news_id_map[row["id"]] = idx

                    e_ids = []
                    if self.config.USE_ENTITIES and isinstance(row["t_ent"], str):
                        found_ids = re.findall(
                            r'"WikidataId":\s*"([^"]+)"', row["t_ent"]
                        )
                        for fid in found_ids:
                            e_ids.append(self.entity_dict.get(fid, 1))

                    e_ids = e_ids[: self.config.MAX_ENTITY_LEN]
                    e_ids += [0] * (self.config.MAX_ENTITY_LEN - len(e_ids))

                    if self.config.USE_LLM:
                        t = self.tokenizer(
                            str(row["title"]),
                            max_length=self.config.MAX_TITLE_LEN,
                            padding="max_length",
                            truncation=True,
                            return_tensors="pt",
                        )
                        self.news_features[idx] = {
                            "text": t["input_ids"].squeeze(0),
                            "mask": t["attention_mask"].squeeze(0),
                            "ents": torch.tensor(e_ids, dtype=torch.long),
                        }
                    else:
                        tokens = re.sub(
                            r"[^\w\s]", "", str(row["title"]).lower()
                        ).split()[: self.config.MAX_TITLE_LEN]

                        for w in tokens:
                            if w not in self.word_dict:
                                self.word_dict[w] = len(self.word_dict)

                        ids = [self.word_dict.get(w, 1) for w in tokens]
                        ids += [0] * (self.config.MAX_TITLE_LEN - len(ids))

                        self.news_features[idx] = {
                            "text": torch.tensor(ids),
                            "mask": (torch.tensor(ids) > 0).long(),
                            "ents": torch.tensor(e_ids, dtype=torch.long),
                        }