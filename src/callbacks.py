# src/callbacks.py
import os
import torch
import pandas as pd
import pytorch_lightning as pl
from rich import print

class GPUMemoryCallback(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            print("GPU Memory tracking started...")

    def on_train_end(self, trainer, pl_module):
        if torch.cuda.is_available():
            max_memory = torch.cuda.max_memory_allocated()
            print(f"\n{'='*40}")
            print(f"Peak GPU Memory Usage: {max_memory / (1024 ** 3):.2f} GB")
            print(f"{'='*40}\n")
        else:
            print("CUDA not available, skipped memory tracking.")

class SaveTestResultsCallback(pl.Callback):
    def __init__(self, filename="test_predictions.csv", output_dir=None):
        super().__init__()
        self.filename = filename
        self.output_dir = output_dir  # Store the specific path
        self.predictions = []
        self.targets = []
        self.imp_ids = [] 

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # ... (Same as before) ...
        preds = outputs["preds"].detach().cpu().numpy()
        targets = outputs["targets"].detach().cpu().numpy()
        self.predictions.extend(preds.flatten().tolist())
        self.targets.extend(targets.flatten().tolist())
        if "imp_id" in outputs:
            self.imp_ids.extend(outputs["imp_id"].detach().cpu().numpy().flatten().tolist())

    def on_test_epoch_end(self, trainer, pl_module):
        # Logic: 
        # 1. Use explicitly provided output_dir (Best)
        # 2. Use trainer logger dir (If available)
        # 3. Fallback to default_root_dir
        
        if self.output_dir:
            save_dir = self.output_dir
        elif trainer.logger and trainer.logger.log_dir:
            save_dir = trainer.logger.log_dir
        else:
            save_dir = trainer.default_root_dir
            
        # Ensure directory exists
        os.makedirs(save_dir, exist_ok=True)
            
        save_path = os.path.join(save_dir, self.filename)

        # ... (DataFrame creation and saving remains the same) ...
        data = {
            "prediction": self.predictions,
            "target": self.targets
        }
        if self.imp_ids:
            data["imp_id"] = self.imp_ids

        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False)
        
        print(f"\n[bold green]Test results saved to:[/bold green] {save_path}")
        
        self.predictions = []
        self.targets = []
        self.imp_ids = []