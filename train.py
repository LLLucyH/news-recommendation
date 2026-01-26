r"""
MIND Training Script
====================

This script handles the end-to-end training and evaluation pipeline for the
MIND News Recommendation model.

The pipeline consists of two distinct phases:
1.  **Multi-GPU Training (DDP)**: The model is trained using Distributed Data Parallel
    (if multiple GPUs are available) to maximize throughput.
2.  **Single-GPU Evaluation**: After training, the process group is torn down, and
    the rank 0 process re-initializes a single-device trainer to run the test set
    evaluation on the best checkpoint.

Usage:
    python train.py --data_root ./data --gpus 2 --batch_size 64
"""

import argparse
import os
import sys

import pytorch_lightning as pl
import torch
import torch.distributed as dist
from pytorch_lightning.callbacks import ModelCheckpoint
from rich import print
from transformers import logging

from src.callbacks import GPUMemoryCallback, SaveTestResultsCallback
from src.config import Config
from src.dataset import MINDDataModule
from src.model import ImpressionClassifier
from src.processor import MindProcessor

logging.set_verbosity_error()


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for training configuration.

    Returns:
        argparse.Namespace: Parsed arguments containing data paths, hyperparameters,
        and model toggles.
    """
    parser = argparse.ArgumentParser(description="MIND News Recommender Training")

    parser.add_argument(
        "--data_root", type=str, default="data", help="Root directory for data"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert_model",
        help="Name of the pretrained model",
    )

    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="Learning rate"
    )
    parser.add_argument(
        "--neg_samples",
        type=int,
        default=2,
        help="Number of negative samples per positive sample",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    parser.add_argument(
        "--val_split",
        type=float,
        default=0.05,
        help="Validation split ratio",
    )

    parser.add_argument(
        "--hidden_dim", type=int, default=256, help="Hidden dimension size"
    )
    parser.add_argument(
        "--num_heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument(
        "--max_title_len", type=int, default=50, help="Maximum title length"
    )
    parser.add_argument(
        "--max_hist_len", type=int, default=30, help="Maximum history length"
    )
    parser.add_argument(
        "--max_ent_len", type=int, default=5, help="Maximum entity list length"
    )

    parser.add_argument(
        "--no_llm",
        action="store_false",
        dest="use_llm",
        help="Disable LLM features",
    )
    parser.add_argument(
        "--no_ent",
        action="store_false",
        dest="use_entities",
        help="Disable entity features",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (use small data subset)",
    )

    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: all available)",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main execution function.

    Steps:
    1.  Initialize configuration and data processor.
    2.  Load entity embeddings and build vocabularies.
    3.  Setup DataModule and Model.
    4.  Run Phase 1: Training (supports DDP).
    5.  Handle DDP teardown (ensure only Rank 0 proceeds).
    6.  Run Phase 2: Testing (Single GPU) using the best checkpoint.
    """
    args = parse_args()

    cfg = Config(
        DATA_ROOT=args.data_root,
        MODEL_NAME=args.model_name,
        MAX_TITLE_LEN=args.max_title_len,
        MAX_HISTORY_LEN=args.max_hist_len,
        MAX_ENTITY_LEN=args.max_ent_len,
        HIDDEN_DIM=args.hidden_dim,
        BATCH_SIZE=args.batch_size,
        NUM_HEADS=args.num_heads,
        LEARNING_RATE=args.lr,
        NEG_SAMPLES=args.neg_samples,
        DROPOUT=args.dropout,
        VAL_SPLIT_RATIO=args.val_split,
        DEBUG=args.debug,
        USE_LLM=args.use_llm,
        USE_ENTITIES=args.use_entities,
    )

    if args.gpus is not None:
        cfg.NUM_DEVICES = args.gpus

    print(
        f"[bold cyan]Config Ready:[/bold cyan] LLM={cfg.USE_LLM}, "
        f"Entities={cfg.USE_ENTITIES}, BatchSize={cfg.BATCH_SIZE}"
    )

    proc = MindProcessor(cfg)

    entity_matrix = proc.load_entity_embeddings()

    print("[bold yellow]Building dictionaries...[/bold yellow]")
    proc.build_dictionaries([cfg.TRAIN_NEWS, cfg.VAL_NEWS])

    dm = MINDDataModule(proc, cfg)

    model = ImpressionClassifier(cfg, len(proc.word_dict), entity_vectors=entity_matrix)

    checkpoint_callback = ModelCheckpoint(
        filename="best-{epoch:03d}",
        monitor="val_auc",
        mode="max",
        save_top_k=1,
        save_last=True,
        verbose=True,
    )

    print(
        f"[bold blue]Starting TRAINING on {cfg.NUM_DEVICES} device(s)...[/bold blue]"
    )

    strategy = "ddp" if cfg.NUM_DEVICES > 1 else "auto"

    trainer = pl.Trainer(
        max_epochs=cfg.EPOCHS,
        accelerator=cfg.DEVICE,
        devices=cfg.NUM_DEVICES,
        strategy=strategy,
        callbacks=[GPUMemoryCallback(), checkpoint_callback],
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=dm)

    # Guard Phase 2 for Rank 0 Only
    # If using DDP, ensure all processes have finished training before proceeding.
    if cfg.NUM_DEVICES > 1 and dist.is_initialized():
        dist.barrier()

    # Non-zero ranks must exit here to prevent them from running Phase 2.
    if trainer.global_rank != 0:
        print(f"[Rank {trainer.global_rank}] Training finished. Exiting process.")
        sys.exit(0)

    # Clean up the DDP process group so the next Trainer doesn't get confused.
    if dist.is_initialized():
        dist.destroy_process_group()

    print(
        "\n[bold magenta]--- Switching to Single GPU for Test Evaluation ---[/bold magenta]"
    )

    train_log_dir = (
        trainer.logger.log_dir if trainer.logger else trainer.default_root_dir
    )
    print(f"Target output directory: {train_log_dir}")

    save_results_callback = SaveTestResultsCallback(
        filename="test_eval_results.csv", output_dir=train_log_dir
    )

    # Create a fresh trainer for evaluation.
    # We strictly use 1 GPU here to avoid DDP overhead during inference/testing.
    eval_trainer = pl.Trainer(
        accelerator=cfg.DEVICE,
        devices=1,
        callbacks=[save_results_callback],
        logger=False,
    )

    best_path = checkpoint_callback.best_model_path
    print(f"Loading best checkpoint from: {best_path}")

    eval_trainer.test(model, datamodule=dm, ckpt_path=best_path)

    print("[bold green]Script finished successfully. Exiting.[/bold green]")
    os._exit(0)


if __name__ == "__main__":
    main()