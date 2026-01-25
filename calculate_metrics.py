import numpy as np
import torch
from sklearn.metrics import roc_auc_score

def dcg_score(y_true, y_score, k=10):
    """Discounted Cumulative Gain"""
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=10):
    """Normalized Discounted Cumulative Gain"""
    best_dcg = dcg_score(y_true, y_true, k)
    if best_dcg == 0:
        return 0.
    return dcg_score(y_true, y_score, k) / best_dcg

def mrr_score(y_true, y_score):
    """Mean Reciprocal Rank"""
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)

def evaluate(model, dataloader, device):
    """
    Evaluates the model on ranking metrics.
    Crucial: Predictions are grouped by Impression to simulate a real Feed.
    """
    model.eval()
    all_auc = []
    all_mrr = []
    all_ndcg5 = []
    all_ndcg10 = []


    with torch.no_grad():
        y_scores = []
        y_true = []