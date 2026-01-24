import torch
from torch.utils.data import Dataset

class MINDDataset(Dataset):
    def __init__(self, behaviors_path, processor, config, mode='train'):
        self.config = config
        self.processor = processor
        self.mode = mode
        
        # Load behaviors
        self.data = pd.read_csv(behaviors_path, sep='\t', 
                                names=['imp_id', 'u_id', 'time', 'history', 'impressions'])
        
        self.samples = self._prepare_samples()

    def _prepare_samples(self):
        samples = []
        for _, row in self.data.iterrows():
            history_indices = self.processor.get_user_history(row['history'])
            
            # Split impressions: "N123-1 N456-0"
            imps = row['impressions'].split()
            positives = [i.split('-')[0] for i in imps if i.split('-')[1] == '1']
            negatives = [i.split('-')[0] for i in imps if i.split('-')[1] == '0']
            
            if self.mode == 'train':
                # Training logic: for every positive, pick K negatives (Negative Sampling)
                for pos in positives:
                    neg_sample = np.random.choice(negatives, self.config.NEG_SAMPLES, replace=True)
                    # Label 1 for positive
                    samples.append((history_indices, self.processor.news_id_map.get(pos, 0), 1.0))
                    # Label 0 for negatives
                    for neg in neg_sample:
                        samples.append((history_indices, self.processor.news_id_map.get(neg, 0), 0.0))
            else:
                # Validation logic: include everything as it is
                for imp in imps:
                    nid, label = imp.split('-')
                    samples.append((history_indices, self.processor.news_id_map.get(nid, 0), float(label)))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hist, candidate, label = self.samples[idx]
        
        # Fetch the title features for history and candidate
        hist_features = np.stack([self.processor.news_features[nid] for nid in hist])
        cand_features = self.processor.news_features[candidate]
        
        return {
            "history": torch.tensor(hist_features, dtype=torch.long),
            "candidate": torch.tensor(cand_features, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float)
        }