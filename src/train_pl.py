import os
import sys
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pathlib import Path
import json

# --- Project Imports ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'src'))

from config import DATASET_CONFIG, MODEL_CONFIG
from dataset.efi_datamodule import EfiDataModule
from models.efi_modelmodule import EfiModelModule

# ---------------------------
# Argument Parsing
# ---------------------------

def parse_args():
    parser = argparse.ArgumentParser('EFI Training / Testing')
    parser.add_argument('--mode', default='train', choices=['train', 'test'], help='Run mode')
    parser.add_argument('--use_cpu', action='store_true', default=False)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--model', type=str, default='dgcnn')
    parser.add_argument('--dataset_config_key', type=str, default='petawawa_cls',
                        choices=DATASET_CONFIG.keys())
    parser.add_argument('--epoch', default=150, type=int)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--num_point', type=int, default=8192)
    parser.add_argument('--log_dir', type=str, default='log', help='Experiment name')
    parser.add_argument('--decay_rate', type=float, default=1e-4)
    parser.add_argument('--process_data', action='store_true', default=False)
    parser.add_argument('--no_fps', action='store_true', help='disable FPS and use uniform sampling')
    parser.add_argument('--wandb_project', type=str, default='EFI_DL', help='Weights & Biases Project Name')
    parser.add_argument('--ddp', action='store_true', default=False, help='Use DDP for multi-GPU training')
    parser.add_argument('--pretrained_ckpt', type=str, default=None, help='Path to pretrained checkpoint')
    
    return parser.parse_args()

# ---------------------------
# W&B Resume Utilities
# ---------------------------

def save_wandb_run_id(exp_dir, run_id):
    path = Path(exp_dir) / "wandb_run.json"
    with open(path, "w") as f:
        json.dump({"run_id": run_id}, f)

def load_wandb_run_id(exp_dir):
    path = Path(exp_dir) / "wandb_run.json"
    if not path.exists():
        raise FileNotFoundError(f"No W&B run file found: {path}")
    with open(path, "r") as f:
        return json.load(f)["run_id"]

# ---------------------------
# Main
# ---------------------------

def main(args):
    # ---------------------------
    # Config Setup
    # ---------------------------
    dataset_cfg = DATASET_CONFIG[args.dataset_config_key]
    model_cfg = MODEL_CONFIG[args.model]

    task = 'classification' if dataset_cfg['num_classes'] > 1 else 'regression'

    if args.pretrained_ckpt == 'None':
        args.pretrained_ckpt = None

    full_cfg = {
        'model': args.model,
        'model_name': args.model,
        'use_fps': (not args.no_fps),
        'num_classes': dataset_cfg['num_classes'],
        'task': task,
        **dataset_cfg,
        **model_cfg
    }

    # ---------------------------
    # Experiment Directory
    # ---------------------------
    exp_root = Path('./log') / args.wandb_project / args.log_dir
    ckpt_dir = exp_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # W&B Logger
    # ---------------------------
    if args.mode == "train":
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            group=task,
            job_type=args.model,
            name=args.log_dir,
            config={**full_cfg, **vars(args)},
        )
        save_wandb_run_id(exp_root, wandb_logger.experiment.id)
    else:
        run_id = load_wandb_run_id(exp_root)
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            group=task,
            job_type="test",
            name=args.log_dir,
            id=run_id,
            resume="must"
        )
    # set seeds for pseudo-random generators
    seed_everything(42, workers=True)
    # ---------------------------
    # Data + Model
    # ---------------------------
    data_module = EfiDataModule(full_cfg, args)

    # Only construct model manually in train
    model_module = None if args.mode == "test" else EfiModelModule(full_cfg, args)

    # ---------------------------
    # Callbacks
    # ---------------------------
    primary_metric = 'val/overall_acc' if task == 'classification' else 'val/loss'
    ckpt_mode = 'max' if task == 'classification' else 'min'
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='best_model',
        monitor=primary_metric,
        mode=ckpt_mode, # Maximize accuracy/R2 or minimize loss
        save_top_k=1,
        verbose=True
    )

    early_stop_callback = EarlyStopping(
        monitor=primary_metric,
        patience=10,
        mode=ckpt_mode
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # OPTIONAL: set up distributed training with multiple GPUs if needed
    if args.ddp:
        n_gpus = torch.cuda.device_count()
        gpu_strategy = DDPStrategy(process_group_backend= "gloo", 
                                   find_unused_parameters=False,
                                   gradient_as_bucket_view=True)
    else:
        n_gpus = 1
        gpu_strategy = 'auto'

    # ---------------------------
    # Trainer
    # ---------------------------
    trainer = pl.Trainer(
        num_nodes=1,
        strategy=gpu_strategy,
        devices=n_gpus,
        max_epochs=args.epoch,
        accelerator='gpu' if torch.cuda.is_available() and not args.use_cpu else 'cpu',
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
        enable_progress_bar=True,
        log_every_n_steps=2,
    )

    # ---------------------------
    # Run
    # ---------------------------
    if args.mode == "train":
        print("\n--- TRAINING ---")
        trainer.fit(model_module, datamodule=data_module)

    else:
        print("\n--- TEST ONLY ---")
        trainer.test(
            ckpt_path=str(ckpt_dir / "best_model.ckpt"),
            datamodule=data_module
        )

# ---------------------------
if __name__ == '__main__':
    args = parse_args()
    main(args)
