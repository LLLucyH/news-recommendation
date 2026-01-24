import pandas as pd
import numpy as np
import collections
import re

class MindProcessor:
    def __init__(self, config):
        self.config = config
        self.word_dict = {"<PAD>": 0, "<OOV>": 1}
        self.news_id_map = {"<PAD>": 0} # Maps NewsID (N123) to integer index
        self.news_features = {} # Stores processed tensors for each news index

    def _tokenize(self, text):
        """Simple tokenizer; in a full version, use NLTK or BERT."""
        if not isinstance(text, str): return []
        tokens = re.sub(r'[^\w\s]', '', text.lower()).split()
        return tokens[:self.config.MAX_TITLE_LEN]

    def build_dictionaries(self, news_path):
        """Processes news.tsv to build vocabulary and news feature lookup."""
        df = pd.read_csv(news_path, sep='\t', names=['id', 'cat', 'subcat', 'title', 'abs', 'url', 'title_ent', 'abs_ent'])
        
        # 1. Build Word Dictionary (Vocab)
        for title in df['title']:
            for word in self._tokenize(title):
                if word not in self.word_dict:
                    self.word_dict[word] = len(self.word_dict)

        # 2. Map News IDs and Convert Titles to Sequences
        for i, row in df.iterrows():
            news_idx = len(self.news_id_map)
            self.news_id_map[row['id']] = news_idx
            
            # Convert title to fixed-length integer sequence
            tokens = self._tokenize(row['title'])
            token_ids = [self.word_dict.get(t, 1) for t in tokens]
            token_ids += [0] * (self.config.MAX_TITLE_LEN - len(token_ids)) # Padding
            
            self.news_features[news_idx] = np.array(token_ids)

        # Add a padding feature for index 0
        self.news_features[0] = np.zeros(self.config.MAX_TITLE_LEN, dtype=int)
        
        print(f"Vocab size: {len(self.word_dict)}")
        print(f"Total News processed: {len(self.news_id_map)}")

    def get_user_history(self, history_str):
        """Converts space-separated History string to list of news indices."""
        if pd.isna(history_str):
            return [0] * self.config.MAX_HISTORY_LEN
        
        hist = history_str.split()
        hist_ids = [self.news_id_map.get(nid, 0) for nid in hist]
        
        # Truncate or Pad
        if len(hist_ids) > self.config.MAX_HISTORY_LEN:
            hist_ids = hist_ids[-self.config.MAX_HISTORY_LEN:]
        else:
            hist_ids = [0] * (self.config.MAX_HISTORY_LEN - len(hist_ids)) + hist_ids
        return hist_ids