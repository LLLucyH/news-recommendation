r"""
Random Search Launcher
======================

This script automates the process of hyperparameter tuning using random search.
It defines a search space for various model parameters (e.g., learning rate,
batch size, dimensions) and submits multiple training jobs to a Slurm cluster
via `sbatch`.

The script assumes the existence of a `train.sh` shell script that accepts
hyperparameters as positional arguments in a specific order.

Usage:
    python random_search.py
"""

import os
import random
import time
from typing import Any, Dict, List

random.seed(0)


def main() -> None:
    """
    Main execution function for random search.

    Steps:
    1.  Define the hyperparameter search space.
    2.  Create the logging directory if it does not exist.
    3.  Iteratively sample hyperparameters and submit Slurm jobs.
    """
    # Number of parallel experiments to launch
    num_experiments = 10

    # Define the search space
    search_space: Dict[str, List[Any]] = {
        "max_title_len": [10, 20, 30, 50],
        "max_hist_len": [10, 20, 30, 50],
        "max_ent_len": [3, 5, 8, 10],
        "hidden_dim": [64, 128, 256],
        "batch_size": [128, 256, 512],
        "lr": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
        "neg_samples": [2, 4, 8],
        "dropout": [0.1, 0.2, 0.3, 0.5],
    }

    if not os.path.exists("logs"):
        os.makedirs("logs")

    for i in range(num_experiments):
        t_len = random.choice(search_space["max_title_len"])
        h_len = random.choice(search_space["max_hist_len"])
        e_len = random.choice(search_space["max_ent_len"])
        h_dim = random.choice(search_space["hidden_dim"])
        bs = random.choice(search_space["batch_size"])
        lr = random.choice(search_space["lr"])
        neg = random.choice(search_space["neg_samples"])
        drp = random.choice(search_space["dropout"])

        # Construct sbatch command
        # Note: The order of arguments must match the variables expected by train.sh
        # Order: MAX_T MAX_H MAX_E HDIM BS LR NEG DRP
        cmd = (
            f"sbatch train.sh "
            f"{t_len} {h_len} {e_len} {h_dim} {bs} {lr} {neg} {drp}"
        )

        print(f"Launching Job {i+1}: {cmd}")
        os.system(cmd)
        time.sleep(1)


if __name__ == "__main__":
    main()