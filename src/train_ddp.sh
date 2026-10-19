#!/bin/bash

EPOCHS=150

LOG_FILE="bash_reg_log.txt"
exec > >(tee -a bash_reg_log.txt) 2>&1

DATASET_KEY="petawawa_reg"

echo "Starting training... ($(date))" | tee -a "$LOG_FILE"

model='ocnn'

lr=0.0001

bs=8

log_dir="peta_reg_${model}_bs${bs}_lr${lr}_3"

echo -e "\n[START $(date)] $model | bs=$bs | lr=$lr" | tee -a "$LOG_FILE"

python train_pl.py \
--model "$model" \
--mode "train" \
--dataset_config_key "$DATASET_KEY" \
--batch_size "$bs" \
--epoch "$EPOCHS" \
--learning_rate "$lr" \
--log_dir "$log_dir" \
--ddp

# sleep 10
pred_ckpt_path="log/efi_dl_workshop/${log_dir}/checkpoints/best_model.ckpt"

python predict.py \
    --model "$model" \
    --dataset_config_key "$DATASET_KEY" \
    --batch_size "$bs" \
    --ckpt_path "$pred_ckpt_path"

echo -e "\nTraining complete. ($(date))" | tee -a "$LOG_FILE"
