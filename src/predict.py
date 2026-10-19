import os
import sys
import argparse
import torch
import pytorch_lightning as pl
import pandas as pd
from pathlib import Path

# --- Project Imports ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'src'))

from config import DATASET_CONFIG, MODEL_CONFIG
from dataset.efi_datamodule import EfiDataModule
from models.efi_modelmodule import EfiModelModule


# -------------------------
# Argument Parsing
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser("EFI Testing & Prediction")

    parser.add_argument('--use_cpu', action='store_true', default=False)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=8)

    parser.add_argument('--model', type=str, default='pointnet')
    parser.add_argument('--dataset_config_key', type=str, default='petawawa_cls',
                        choices=DATASET_CONFIG.keys())

    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to best_model.ckpt')
    parser.add_argument('--out_csv', type=str, default='predictions.csv',
                        help='Output CSV file')
    parser.add_argument('--num_point', type=int, default=8192, help='Point Number')
    parser.add_argument('--no_fps', action='store_true', help='disable FPS and use uniform sampling')
    parser.add_argument('--pretrained_ckpt', type=str, default=None, help='Path to pretrained checkpoint')

    return parser.parse_args()


# -------------------------
# Main
# -------------------------
def main(args):

    # 1. Config setup
    dataset_cfg = DATASET_CONFIG[args.dataset_config_key]
    model_cfg = MODEL_CONFIG[args.model]

    task = 'classification' if dataset_cfg['num_classes'] > 1 else 'regression'

    full_cfg = {
        'model_name': args.model,
        'num_classes': dataset_cfg['num_classes'],
        'task': task,
        'use_fps': (not args.no_fps),
        **dataset_cfg,
        **model_cfg
    }

    # 2. Data + Model
    data_module = EfiDataModule(full_cfg, args)
    model = EfiModelModule.load_from_checkpoint(
        args.ckpt_path,
        cfg=full_cfg,
        args=args
    )

    # 3. Trainer (NO logger needed)
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() and not args.use_cpu else 'cpu',
        devices=[int(g) for g in args.gpu.split(',')] if ',' in args.gpu else 1,
        enable_progress_bar=True,
        logger=False
    )

    # 4. Run prediction
    print("Running prediction...")
    outputs = trainer.predict(model, dataloaders=data_module)

    # 5. Parse outputs → CSV
    records = []

    if task == "classification":
        for o in outputs:
            for pid, gt, pred in zip(
                o["plot_id"],
                o["gt"],
                o["pred"]
            ):
                records.append({
                    "plot_id": pid,
                    "dom_sp_type": int(gt.item()),
                    "pred_dom_sp_type": int(pred.item())
                })

    else:  # regression
        for o in outputs:
            for pid, gt, pred in zip(
                o["plot_id"],
                o["gt"],
                o["pred"]
            ):
                records.append({
                    "plot_id": pid,
                    "total_agb_z": float(gt.item()),
                    "total_agb_z_pred": float(pred.item())
                })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(os.path.dirname(args.ckpt_path), args.out_csv), index=False)

    print(f"Predictions saved to: {os.path.join(os.path.dirname(args.ckpt_path), args.out_csv)}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
