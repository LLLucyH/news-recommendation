r"""
MIND Baseline Training Script
=============================

This script trains a baseline Logistic Regression model for the MIND news recommendation
task. It supports two feature extraction methods:

1.  **Entity Embeddings**: Uses the average of pre-trained Knowledge Graph embeddings
    associated with entities in the news title.
2.  **TF-IDF + SVD**: Uses Latent Semantic Analysis (LSA) on the news titles to
    generate dense vectors.

The model represents a user as the average vector of their history and a candidate
news item as its own vector. The features passed to the classifier are the concatenation
of the User Vector and the Candidate Vector.

Usage:
    python train_baseline.py --method tfidf --svd_dim 100
    python train_baseline.py --method entity
"""

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from rich import print

from src.config import Config
from src.processor import MindProcessor


def get_entity_vector_map(
    processor: MindProcessor,
) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Extract mean entity vectors for all news items using the MindProcessor.

    Iterates through the news dictionary in the processor. For each news item,
    it retrieves the associated entity IDs, looks up their pre-trained embeddings,
    and computes the mean vector.

    Args:
        processor (MindProcessor): Initialized processor with loaded entity embeddings
            and news dictionaries.

    Returns:
        Tuple[Dict[str, np.ndarray], int]:
            - A dictionary mapping News ID (str) to its dense vector (np.ndarray).
            - The dimension of the embeddings.
    """
    dim = processor.config.ENTITY_EMB_DIM
    news_vec_map = {}

    print("Building news vector cache from Entities...")
    for nid, idx in tqdm(processor.news_id_map.items(), desc="Processing Entities"):
        if nid == "<PAD>":
            news_vec_map[nid] = np.zeros(dim, dtype=np.float32)
            continue

        feat = processor.news_features.get(idx)
        if feat is None:
            news_vec_map[nid] = np.zeros(dim, dtype=np.float32)
            continue

        ent_ids = feat["ents"].numpy()
        # Filter out padding (0) and OOV (1) if desired, or just padding
        vecs = [processor.entity_vectors[eid].numpy() for eid in ent_ids if eid > 1]

        if not vecs:
            news_vec_map[nid] = np.zeros(dim, dtype=np.float32)
        else:
            news_vec_map[nid] = np.mean(vecs, axis=0).astype(np.float32)

    return news_vec_map, dim


def get_tfidf_vector_map(
    cfg: Config, n_components: int = 100
) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Calculate TF-IDF + SVD (LSA) vectors for all news titles.

    1.  Loads all news titles from training and validation sets.
    2.  Computes a TF-IDF matrix (max 10k features).
    3.  Reduces dimensionality using TruncatedSVD to `n_components`.

    Args:
        cfg (Config): Configuration object containing paths to news files.
        n_components (int, optional): Target dimension for SVD. Defaults to 100.

    Returns:
        Tuple[Dict[str, np.ndarray], int]:
            - A dictionary mapping News ID (str) to its dense vector (np.ndarray).
            - The dimension of the resulting vectors.
    """
    print("Loading news for TF-IDF...")
    news_train = pd.read_csv(
        cfg.TRAIN_NEWS,
        sep="\t",
        names=["id", "cat", "sub", "title", "abs", "url", "t_ent", "a_ent"],
    )
    news_val = pd.read_csv(
        cfg.VAL_NEWS,
        sep="\t",
        names=["id", "cat", "sub", "title", "abs", "url", "t_ent", "a_ent"],
    )
    
    all_news = pd.concat([news_train, news_val]).drop_duplicates("id")

    titles = all_news["title"].fillna("").tolist()
    news_ids = all_news["id"].tolist()

    print(f"Fitting TF-IDF and SVD (dim={n_components})...")
    tfidf = TfidfVectorizer(stop_words="english", max_features=10000)
    tfidf_matrix = tfidf.fit_transform(titles)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    low_dim_vectors = svd.fit_transform(tfidf_matrix)

    news_vec_map = {
        nid: vec.astype(np.float32) for nid, vec in zip(news_ids, low_dim_vectors)
    }
    news_vec_map["<PAD>"] = np.zeros(n_components, dtype=np.float32)

    return news_vec_map, n_components


def prepare_features(
    df: pd.DataFrame,
    news_vec_map: Dict[str, np.ndarray],
    dim: int,
    limit: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Convert behavior dataframe into feature matrices (X) and labels (y).

    For each impression:
    1.  **User Vector**: Average of vectors for all news in the user's history.
    2.  **Candidate Vector**: Vector for the specific candidate news item.
    3.  **Feature**: Concatenation of [User Vector, Candidate Vector].

    Args:
        df (pd.DataFrame): Behavior dataframe containing history and impressions.
        news_vec_map (Dict[str, np.ndarray]): Mapping from News ID to dense vector.
        dim (int): Dimension of the single news vector.
        limit (int, optional): Limit the number of rows processed (for debugging).

    Returns:
        Tuple[np.ndarray, np.ndarray, List[str]]:
            - X: Feature matrix of shape (n_samples, dim * 2).
            - y: Label array of shape (n_samples,).
            - imp_ids: List of impression IDs corresponding to each row in X.
    """
    if limit:
        df = df.head(limit)

    print("Calculating sample size...")
    total_samples = sum(len(str(row).split()) for row in df["impressions"])

    X = np.zeros((total_samples, dim * 2), dtype=np.float32)
    y = np.zeros(total_samples, dtype=np.int32)
    imp_ids = []

    zero_vec = np.zeros(dim, dtype=np.float32)
    curr_idx = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating Features"):
        # 1. User Vector (Mean of history)
        hist_news = str(row["history"]).split() if pd.notna(row["history"]) else []
        hist_vecs = [news_vec_map.get(nid, zero_vec) for nid in hist_news]
        user_vec = np.mean(hist_vecs, axis=0) if hist_vecs else zero_vec

        # 2. Candidate News Vectors
        impressions = str(row["impressions"]).split()
        for imp in impressions:
            news_id, label = imp.split("-")
            item_vec = news_vec_map.get(news_id, zero_vec)

            # Concatenate [User_Vector, Item_Vector]
            X[curr_idx, :dim] = user_vec
            X[curr_idx, dim:] = item_vec
            y[curr_idx] = int(label)
            imp_ids.append(row["imp_id"])
            curr_idx += 1

    return X, y, imp_ids


def main() -> None:
    """
    Main execution function for baseline training.

    Steps:
    1.  Parse arguments (method selection, dimensions).
    2.  Build or load news vectors (Entity or TF-IDF).
    3.  Load behavior data (Train/Val).
    4.  Generate feature matrices (X, y).
    5.  Train Logistic Regression.
    6.  Evaluate AUC on validation set.
    7.  Save predictions to CSV.
    """
    parser = argparse.ArgumentParser(description="MIND Baseline: Entity vs TF-IDF")
    parser.add_argument(
        "--method",
        type=str,
        choices=["entity", "tfidf"],
        default="tfidf",
        help="Choose the feature extraction method",
    )
    parser.add_argument(
        "--train_limit",
        type=int,
        default=None,
        help="Limit training samples for speed",
    )
    parser.add_argument(
        "--svd_dim", type=int, default=100, help="Dimension for TF-IDF SVD"
    )
    args = parser.parse_args()

    cfg = Config()
    cfg.USE_LLM = False

    if args.method == "entity":
        print("--- Running Entity-based Baseline ---")
        processor = MindProcessor(cfg)
        processor.load_entity_embeddings()
        processor.build_dictionaries([cfg.TRAIN_NEWS, cfg.VAL_NEWS])
        news_vec_map, dim = get_entity_vector_map(processor)
    else:
        print("--- Running TF-IDF-based Baseline ---")
        news_vec_map, dim = get_tfidf_vector_map(cfg, n_components=args.svd_dim)

    print("Loading behavior data...")
    train_df = pd.read_csv(
        cfg.TRAIN_BEHAVIORS,
        sep="\t",
        names=["imp_id", "u_id", "time", "history", "impressions"],
    )
    val_df = pd.read_csv(
        cfg.VAL_BEHAVIORS,
        sep="\t",
        names=["imp_id", "u_id", "time", "history", "impressions"],
    )

    print("Processing Training Data...")
    X_train, y_train, _ = prepare_features(
        train_df, news_vec_map, dim, limit=args.train_limit
    )

    print("Processing Validation Data...")
    X_test, y_test, test_imp_ids = prepare_features(val_df, news_vec_map, dim)

    print(f"Training Logistic Regression on {len(X_train)} samples...")
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", n_jobs=-1)
    clf.fit(X_train, y_train)

    print("Evaluating...")
    preds = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    print(f"\n[{args.method.upper()}] Baseline AUC: {auc:.4f}")

    os.makedirs("baseline_results", exist_ok=True)
    output_path = f"baseline_results/{args.method}_baseline_results.csv"
    output_df = pd.DataFrame(
        {"imp_id": test_imp_ids, "prediction": preds, "target": y_test}
    )
    output_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()