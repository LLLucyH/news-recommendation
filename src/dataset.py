import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class MINDDataset(Dataset):
    def __init__(self, df, processor, config, mode='train'):
        self.config, self.processor, self.mode = config, processor, mode
        self.data = df
        self.samples = self._prepare_samples()

    def _prepare_samples(self):
        samples = []
        for _, row in self.data.iterrows():
            iid = int(row['imp_id']) 
            hist = str(row['history']).split()
            hist_ids = [self.processor.news_id_map.get(nid, 0) for nid in hist][-self.config.MAX_HISTORY_LEN:]
            hist_ids = [0] * (self.config.MAX_HISTORY_LEN - len(hist_ids)) + hist_ids
            imps = str(row['impressions']).split()
            
            if self.mode == 'train':
                pos = [i.split('-')[0] for i in imps if i.split('-')[1] == '1']
                neg = [i.split('-')[0] for i in imps if i.split('-')[1] == '0']
                for p in pos:
                    neg_s = np.random.choice(neg, self.config.NEG_SAMPLES, replace=True) if neg else []
                    samples.append((iid, hist_ids, self.processor.news_id_map.get(p, 0), 1.0))
                    for n in neg_s: 
                        samples.append((iid, hist_ids, self.processor.news_id_map.get(n, 0), 0.0))
            else:
                for i in imps:
                    nid, lbl = i.split('-')
                    samples.append((iid, hist_ids, self.processor.news_id_map.get(nid, 0), float(lbl)))
        return samples

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        iid, h_ids, c_id, label = self.samples[idx]
        
        def get_feats(nids):
            return {
                'text': torch.stack([self.processor.news_features[i]['text'] for i in nids]),
                'mask': torch.stack([self.processor.news_features[i]['mask'] for i in nids]),
                'ents': torch.stack([self.processor.news_features[i]['ents'] for i in nids])
            }
        
        h_feats = get_feats(h_ids)
        c_feat = self.processor.news_features[c_id]
        
        return {
            "imp_id": torch.tensor(iid, dtype=torch.long),
            "h_text": h_feats['text'], "h_mask": h_feats['mask'], "h_ents": h_feats['ents'],
            "c_text": c_feat['text'], "c_mask": c_feat['mask'], "c_ents": c_feat['ents'],
            "label": torch.tensor(label, dtype=torch.float)
        }

class MINDDataModule(pl.LightningDataModule):
    def __init__(self, processor, config):
        super().__init__()
        self.processor = processor
        self.config = config

    def setup(self, stage=None):
        train_full = pd.read_csv(self.config.TRAIN_BEHAVIORS, sep='\t', names=['imp_id', 'u_id', 'time', 'history', 'impressions'])
        if self.config.DEBUG: train_full = train_full.head(self.config.DEBUG_SAMPLES)
        train_df, val_df = train_test_split(train_full, test_size=self.config.VAL_SPLIT_RATIO, random_state=42)
        
        test_df = pd.read_csv(self.config.VAL_BEHAVIORS, sep='\t', names=['imp_id', 'u_id', 'time', 'history', 'impressions'])
        if self.config.DEBUG: test_df = test_df.head(100)

        self.train_ds = MINDDataset(train_df, self.processor, self.config, 'train')
        self.val_ds = MINDDataset(val_df, self.processor, self.config, 'val')
        self.test_ds = MINDDataset(test_df, self.processor, self.config, 'test')

    def train_dataloader(self): return DataLoader(self.train_ds, batch_size=self.config.BATCH_SIZE, shuffle=True)
    def val_dataloader(self): return DataLoader(self.val_ds, batch_size=self.config.BATCH_SIZE)
    def test_dataloader(self): return DataLoader(self.test_ds, batch_size=self.config.BATCH_SIZE)