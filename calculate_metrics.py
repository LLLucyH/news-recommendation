r"""
Metrics Evaluation & Bootstrap Script
=====================================

This script evaluates the performance of trained recommendation models by analyzing
prediction CSV files. It calculates standard ranking metrics and employs statistical
bootstrapping to generate 95% Confidence Intervals (CI) for robust comparison.

The script is designed for high-throughput evaluation using vectorized operations
for per-impression metrics and parallel processing for global metrics.

Methodology
-----------
1.  **Data Loading**: Reads CSVs containing `target`, `prediction`, and `imp_id`.
2.  **Preprocessing**: Filters out invalid impressions (those with all positive or
    all negative targets) to ensure ranking metrics are mathematically defined.
3.  **Metric Calculation**:
    *   **AUC**: Area Under the ROC Curve (calculated globally).
    *   **nDCG@K**: Normalized Discounted Cumulative Gain at rank K (averaged per impression).
    *   **MRR**: Mean Reciprocal Rank (averaged per impression).
4.  **Bootstrapping**:
    *   **Ranking Metrics**: Uses vectorized NumPy broadcasting to resample impression
        scores 1000 times instantly.
    *   **Global AUC**: Uses parallel processing (Joblib) to resample the global
        prediction pool and recompute ROC AUC.
    *   **Confidence Intervals**: Reports the mean and the 2.5th to 97.5th percentiles.

Output
------
*   Real-time progress updates and metric printing.
*   A formatted summary table printed to the console at the end.
*   A CSV file (`metrics_summary.csv`) containing the final results.
"""

import os
from typing import Dict, Union

import numpy as np
import pandas as pd
import joblib
import sklearn.metrics
import rich
from rich.table import Table
from tqdm import tqdm

console = rich.console.Console()


def get_metric_arrays(file_path: str) -> Union[Dict[str, np.ndarray], str]:
    """
    Loads prediction data and calculates raw metric arrays for every valid impression.

    This function performs the heavy lifting of data processing:
    1.  Reads the raw CSV.
    2.  Filters out impressions that lack variance (all 0s or all 1s).
    3.  Sorts predictions by impression ID and score (descending).
    4.  Computes MRR and nDCG scores for each impression individually.

    Args:
        file_path (str): Path to the CSV file containing columns 'target',
            'prediction', and 'imp_id'.

    Returns:
        Union[Dict[str, np.ndarray], str]: A dictionary containing:
            - 'y_true': Global array of targets (for AUC).
            - 'y_score': Global array of predictions (for AUC).
            - 'MRR': Array of MRR scores (one per impression).
            - 'nDCG@5': Array of nDCG@5 scores (one per impression).
            - 'nDCG@10': Array of nDCG@10 scores (one per impression).
            Returns an error string if the file is invalid or columns are missing.
    """
    try:
        df = pd.read_csv(file_path)
        required = ["target", "prediction", "imp_id"]
        if not all(col in df.columns for col in required):
            return "Column Error"

        y_true_global = df["target"].values
        y_score_global = df["prediction"].values

        g = df.groupby("imp_id")["target"]
        n_pos = g.transform("sum")
        n_count = g.transform("count")
        valid_mask = (n_pos > 0) & (n_pos < n_count)
        df_clean = df[valid_mask].copy()

        if df_clean.empty:
            return "No valid groups"

        df_clean.sort_values(by=["imp_id", "prediction"], ascending=[True, False], inplace=True)
        df_clean["rank"] = df_clean.groupby("imp_id").cumcount() + 1

        g_clean = df_clean.groupby("imp_id")
        n_pos_group = g_clean["target"].sum()

        first_pos_rank = df_clean[df_clean["target"] == 1].groupby("imp_id")["rank"].min()
        mrr_scores = (1.0 / first_pos_rank).reindex(n_pos_group.index, fill_value=0.0).values

        def get_ndcg_values(k):
            df_k = df_clean[df_clean["rank"] <= k].copy()
            df_k["dcg"] = df_k["target"] / np.log2(df_k["rank"] + 1)
            actual_dcg = df_k.groupby("imp_id")["dcg"].sum().reindex(n_pos_group.index, fill_value=0)
            
            max_pos = int(n_pos_group.max())
            discounts = 1.0 / np.log2(np.arange(1, max_pos + 2) + 1)
            idcg_lookup = np.cumsum(discounts)
            
            clipped_n_pos = n_pos_group.clip(upper=k).astype(int)
            ideal_dcg = clipped_n_pos.map(lambda x: idcg_lookup[x-1] if x > 0 else 1.0)
            
            return (actual_dcg / ideal_dcg).values

        return {
            "y_true": y_true_global,
            "y_score": y_score_global,
            "MRR": mrr_scores,
            "nDCG@5": get_ndcg_values(5),
            "nDCG@10": get_ndcg_values(10),
        }

    except Exception as e:
        return f"Error: {str(e)}"


def compute_bootstrap_ci(values: np.ndarray, n_bootstrap: int = 1000) -> str:
    """
    Computes the 95% Confidence Interval for a metric using vectorized bootstrapping.

    This function uses NumPy broadcasting to generate `n_bootstrap` resamples
    instantly, avoiding slow Python loops. It calculates the mean of each
    resample and determines the 2.5th and 97.5th percentiles.

    Args:
        values (np.ndarray): Array of metric scores (e.g., one MRR score per impression).
        n_bootstrap (int): Number of bootstrap iterations. Defaults to 1000.

    Returns:
        str: A formatted string "Mean (Lower-Upper)".
    """
    rng = np.random.default_rng(42)
    n = len(values)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    means = values[indices].mean(axis=1)
    
    mean_val = np.mean(means)
    lower = np.percentile(means, 2.5)
    upper = np.percentile(means, 97.5)
    
    return f"{mean_val:.4f} ({lower:.4f}-{upper:.4f})"


def _auc_worker(y_true: np.ndarray, y_score: np.ndarray, seed: int) -> Union[float, None]:
    """
    Internal worker function for a single bootstrap iteration of Global AUC.

    Args:
        y_true (np.ndarray): True binary labels.
        y_score (np.ndarray): Prediction scores.
        seed (int): Random seed for reproducibility.

    Returns:
        Union[float, None]: The ROC AUC score for the resampled data, or None
        if the resample contains only one class.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y_true), size=len(y_true))
    yt_bs = y_true[idx]
    if len(np.unique(yt_bs)) < 2: 
        return None
    return sklearn.metrics.roc_auc_score(yt_bs, y_score[idx])


def compute_auc_ci(y_true: np.ndarray, y_score: np.ndarray, n_bootstrap: int = 1000) -> str:
    """
    Computes the 95% Confidence Interval for Global AUC using parallel processing.

    Since `roc_auc_score` is computationally expensive and cannot be easily
    vectorized like simple averaging, this function distributes the bootstrap
    iterations across multiple CPU cores using `joblib`. It uses `tqdm` to
    display a progress bar.

    Args:
        y_true (np.ndarray): Global array of true binary labels.
        y_score (np.ndarray): Global array of prediction scores.
        n_bootstrap (int): Number of bootstrap iterations. Defaults to 1000.

    Returns:
        str: A formatted string "Mean (Lower-Upper)", or "N/A" if calculation fails.
    """
    rng = np.random.default_rng(42)
    seeds = rng.integers(0, 10**9, size=n_bootstrap)
    
    results = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(_auc_worker)(y_true, y_score, s) 
        for s in tqdm(seeds, desc="    Global AUC Bootstrap", leave=False, ncols=80)
    )
    aucs = [r for r in results if r is not None]
    
    if not aucs: 
        return "N/A"

    mean_val = np.mean(aucs)
    lower = np.percentile(aucs, 2.5)
    upper = np.percentile(aucs, 97.5)
    
    return f"{mean_val:.4f} ({lower:.4f}-{upper:.4f})"


def main() -> None:
    """
    Main execution entry point.

    1. Defines the list of model versions and their file paths.
    2. Iterates through each model, loading data and computing metrics.
    3. Displays real-time progress and intermediate results.
    4. Displays a final rich-formatted table of results.
    5. Saves the summary data to 'metrics_summary.csv'.
    """
    N_BOOTSTRAP = 100
    OUTPUT_CSV = "metrics_summary.csv"
    
    models = [
        {"label": "TF-IDF Logistic Regression", "path": "baseline_results/tfidf_baseline_results.csv"},
        {"label": "Graph Embedding Logistic Regression", "path": "baseline_results/entity_baseline_results.csv"},
        {"label": "Impression Clf (Not Pretrained, No Graph Embeddings)",    "path": "lightning_logs/version_4005796/test_eval_results.csv"},
        {"label": "Impression Clf (Not Pretrained, With Graph Embeddings)",      "path": "lightning_logs/version_4005791/test_eval_results.csv"},
        {"label": "Impression Clf (Pretrained, No Graph Embeddings)", "path": "lightning_logs/version_4005792/test_eval_results.csv"},
        {"label": "Impression Clf (Pretrained, With Graph Embeddings)",  "path": "lightning_logs/version_4005793/test_eval_results.csv"},
    ]

    table = rich.table.Table(title=f"Model Performance Summary (Bootstrap n={N_BOOTSTRAP})")
    table.add_column("Model Variant", style="cyan", no_wrap=True)
    table.add_column("AUC", justify="center")
    table.add_column("nDCG@5", justify="center")
    table.add_column("nDCG@10", justify="center")
    table.add_column("MRR", justify="center")

    csv_data = []

    console.print(f"[bold]Starting Evaluation with {N_BOOTSTRAP} bootstrap iterations...[/bold]")
    console.print("=" * 80)

    for model in models:
        label = model["label"]
        path = model["path"]
        
        console.print(f"\n[bold blue]Processing: {label}[/bold blue]")
        
        if not os.path.exists(path):
            console.print(f"  [red]File Not Found:[/red] {path}")
            table.add_row(label, "File Not Found", "-", "-", "-")
            continue

        console.print("  Loading and pre-calculating metrics...")
        data = get_metric_arrays(path)
        
        if isinstance(data, str):
            console.print(f"  [red]Error:[/red] {data}")
            table.add_row(label, f"Error: {data}", "-", "-", "-")
            continue

        auc_str = compute_auc_ci(data["y_true"], data["y_score"], N_BOOTSTRAP)
        
        ndcg5_str = compute_bootstrap_ci(data["nDCG@5"], N_BOOTSTRAP)
        ndcg10_str = compute_bootstrap_ci(data["nDCG@10"], N_BOOTSTRAP)
        mrr_str = compute_bootstrap_ci(data["MRR"], N_BOOTSTRAP)

        console.print(f"  [green]AUC:[/green]     {auc_str}")
        console.print(f"  [green]nDCG@5:[/green]  {ndcg5_str}")
        console.print(f"  [green]nDCG@10:[/green] {ndcg10_str}")
        console.print(f"  [green]MRR:[/green]     {mrr_str}")

        table.add_row(label, auc_str, ndcg5_str, ndcg10_str, mrr_str)

        csv_data.append({
            "Model": label,
            "AUC": auc_str,
            "nDCG@5": ndcg5_str,
            "nDCG@10": ndcg10_str,
            "MRR": mrr_str
        })

    console.print("\n" + "=" * 80)
    console.print(table)
    
    if csv_data:
        pd.DataFrame(csv_data).to_csv(OUTPUT_CSV, index=False)
        console.print(f"\n[bold]Results saved to:[/bold] {OUTPUT_CSV}")


if __name__ == "__main__":
    main()