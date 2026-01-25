import subprocess
import torch
import pandas as pd
import numpy as np
import re
import os
from tqdm import tqdm
from transformers import AutoTokenizer
from rich import print

class MindProcessor:
    def __init__(self, config):
        self.config = config
        self.word_dict = {"<PAD>": 0, "<OOV>": 1}
        self.entity_dict = {"<PAD>": 0, "<OOV>": 1}
        self.news_id_map = {"<PAD>": 0} 
        
        self.news_features = {0: {
            'text': torch.zeros(config.MAX_TITLE_LEN, dtype=torch.long), 
            'mask': torch.zeros(config.MAX_TITLE_LEN, dtype=torch.long),
            'ents': torch.zeros(config.MAX_ENTITY_LEN, dtype=torch.long)
        }}
        
        if config.USE_LLM:
            self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, local_files_only=True)
        
        self.entity_vectors = None

    def load_entity_embeddings(self):
        if not self.config.USE_ENTITIES or not os.path.exists(self.config.ENTITY_EMB_PATH):
            print(f"Entity embeddings not found at {self.config.ENTITY_EMB_PATH} or disabled.")
            return None
        
        print("Loading entity embeddings...")
        vecs = []
        vecs.append(np.zeros(self.config.ENTITY_EMB_DIM)) # PAD
        vecs.append(np.random.normal(size=self.config.ENTITY_EMB_DIM)) # OOV
        
        total_lines = int(subprocess.check_output(['wc', '-l', self.config.ENTITY_EMB_PATH]).split()[0])
        with open(self.config.ENTITY_EMB_PATH, 'r') as f:
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

    def build_dictionaries(self, news_paths):
        for path in news_paths:
            if not os.path.exists(path):
                print(f"Warning: News file not found at {path}")
                continue

            df = pd.read_csv(path, sep='\t', names=['id', 'cat', 'sub', 'title', 'abs', 'url', 't_ent', 'a_ent'])
            for _, row in tqdm(df.iterrows(), total=len(df)):
                if row['id'] not in self.news_id_map:
                    idx = len(self.news_id_map); self.news_id_map[row['id']] = idx
                    
                    e_ids = []
                    if self.config.USE_ENTITIES and isinstance(row['t_ent'], str):
                        found_ids = re.findall(r'"WikidataId":\s*"([^"]+)"', row['t_ent'])
                        for fid in found_ids:
                            e_ids.append(self.entity_dict.get(fid, 1))
                    
                    e_ids = e_ids[:self.config.MAX_ENTITY_LEN]
                    e_ids += [0] * (self.config.MAX_ENTITY_LEN - len(e_ids))
                    
                    if self.config.USE_LLM:
                        t = self.tokenizer(str(row['title']), max_length=self.config.MAX_TITLE_LEN, padding='max_length', truncation=True, return_tensors='pt')
                        self.news_features[idx] = {
                            'text': t['input_ids'].squeeze(0), 
                            'mask': t['attention_mask'].squeeze(0),
                            'ents': torch.tensor(e_ids, dtype=torch.long)
                        }
                    else:
                        tokens = re.sub(r'[^\w\s]', '', str(row['title']).lower()).split()[:self.config.MAX_TITLE_LEN]
                        for w in tokens: 
                            if w not in self.word_dict: self.word_dict[w] = len(self.word_dict)
                        ids = [self.word_dict.get(w, 1) for w in tokens]
                        ids += [0] * (self.config.MAX_TITLE_LEN - len(ids))
                        self.news_features[idx] = {
                            'text': torch.tensor(ids), 
                            'mask': (torch.tensor(ids)>0).long(),
                            'ents': torch.tensor(e_ids, dtype=torch.long)
                        }