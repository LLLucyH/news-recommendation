r"""
Callbacks module
================

This module defines custom PyTorch Lightning callbacks used for monitoring system
resources and handling test phase outputs.

The callbacks included are:
1.  **GPUMemoryCallback**: Tracks and logs the peak GPU memory usage during training.
2.  **SaveTestResultsCallback**: Aggregates predictions and targets during the testing
    phase and saves them to a CSV file for offline analysis.

---

### **Core Attributes**
The callbacks interact with the Trainer and LightningModule states.

| **Class**                  | **Description** |
|----------------------------|-----------------|
| `GPUMemoryCallback`        | Resets CUDA stats at start and logs peak usage at end. |
| `SaveTestResultsCallback`  | Collects batch outputs and saves a CSV of predictions. |

"""

import os
from typing import Any, Dict, List, Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from rich import print


class GPUMemoryCallback(pl.Callback):
    """
    Callback to track and log peak GPU memory usage.

    This callback resets the CUDA memory peak statistics at the beginning of training
    and prints the maximum memory allocated upon training completion.
    """

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Reset CUDA memory statistics when training starts.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning trainer instance.
            pl_module (pl.LightningModule): The model module.
        """
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            print("GPU Memory tracking started...")

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Log the peak GPU memory usage when training ends.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning trainer instance.
            pl_module (pl.LightningModule): The model module.
        """
        if torch.cuda.is_available():
            max_memory = torch.cuda.max_memory_allocated()
            print(f"\n{'='*40}")
            print(f"Peak GPU Memory Usage: {max_memory / (1024 ** 3):.2f} GB")
            print(f"{'='*40}\n")
        else:
            print("CUDA not available, skipped memory tracking.")


class SaveTestResultsCallback(pl.Callback):
    """
    Callback to save test predictions to a CSV file.

    Attributes:
        filename (str): Name of the output CSV file.
        output_dir (str, optional): Specific directory to save results.
        predictions (List[float]): Accumulated list of predicted scores.
        targets (List[float]): Accumulated list of ground truth labels.
        imp_ids (List[int]): Accumulated list of impression IDs.
    """

    def __init__(
        self, filename: str = "test_predictions.csv", output_dir: Optional[str] = None
    ):
        """
        Initialize the SaveTestResultsCallback.

        Args:
            filename (str, optional): Output filename. Defaults to "test_predictions.csv".
            output_dir (str, optional): Directory to save the file. If None, logic in
                `on_test_epoch_end` determines the path. Defaults to None.
        """
        super().__init__()
        self.filename = filename
        self.output_dir = output_dir
        self.predictions: List[float] = []
        self.targets: List[float] = []
        self.imp_ids: List[int] = []

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict[str, torch.Tensor],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """
        Aggregate results from a single test batch.

        Extracts predictions, targets, and optional impression IDs from the model outputs,
        detaches them from the graph, moves them to CPU, and extends the internal lists.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning trainer instance.
            pl_module (pl.LightningModule): The model module.
            outputs (Dict[str, torch.Tensor]): Outputs from `test_step`.
            batch (Any): The input batch.
            batch_idx (int): Index of the batch.
            dataloader_idx (int, optional): Index of the dataloader. Defaults to 0.
        """
        preds = outputs["preds"].detach().cpu().numpy()
        targets = outputs["targets"].detach().cpu().numpy()
        self.predictions.extend(preds.flatten().tolist())
        self.targets.extend(targets.flatten().tolist())

        if "imp_id" in outputs:
            self.imp_ids.extend(
                outputs["imp_id"].detach().cpu().numpy().flatten().tolist()
            )

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Save aggregated results to CSV at the end of the test epoch.

        Determines the save directory based on the following priority:
        1.  `self.output_dir` (if explicitly provided).
        2.  `trainer.logger.log_dir` (if a logger is active).
        3.  `trainer.default_root_dir` (fallback).

        After saving, the internal lists are cleared to prevent memory leaks or
        duplication in subsequent runs.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning trainer instance.
            pl_module (pl.LightningModule): The model module.
        """
        if self.output_dir:
            save_dir = self.output_dir
        elif trainer.logger and trainer.logger.log_dir:
            save_dir = trainer.logger.log_dir
        else:
            save_dir = trainer.default_root_dir

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, self.filename)

        data = {"prediction": self.predictions, "target": self.targets}
        if self.imp_ids:
            data["imp_id"] = self.imp_ids

        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False)

        print(f"\n[bold green]Test results saved to:[/bold green] {save_path}")

        self.predictions = []
        self.targets = []
        self.imp_ids = []