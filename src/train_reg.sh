#!/bin/bash

SLEEP_DURATION=10  # seconds
EPOCHS=100

LOG_FILE="bash_reg_log.txt"
exec > >(tee -a bash_reg_log.txt) 2>&1

DATASET_KEY="petawawa_reg"

echo "Starting training... ($(date))" | tee -a "$LOG_FILE"

models=(
  ocnn
  dgcnn
)

batch_sizes=(8 16)
learning_rates=(0.0001 0.001)
pretrained_ckpt=('./pretrained_ckpt/ocnn_nbk_idXM6D.ckpt' 'None')

for model in "${models[@]}"; do
  for ckpt in "${pretrained_ckpt[@]}"; do
    for bs in "${batch_sizes[@]}"; do
      for lr in "${learning_rates[@]}"; do

        # format lr tag
        if [[ "$lr" == "0.0001" ]]; then
          lr_tag="e4"
        elif [[ "$lr" == "0.001" ]]; then
          lr_tag="e3"
        fi

        # Format ckpt tag
        if [[ "$ckpt" != "None" ]]; then
          ckpt_name=$(basename "$ckpt")
          ckpt_name="${ckpt_name%.*}"
        else
          ckpt_name="no_ckpt"
        fi

        log_dir="peta_reg_${model}_bs${bs}_lr${lr_tag}_ckpt_${ckpt_name}"

        echo -e "\n[START $(date)] $model | bs=$bs | lr=$lr | ckpt=$ckpt" | tee -a "$LOG_FILE"

        python train_pl.py \
          --model "$model" \
          --mode "train" \
          --dataset_config_key "$DATASET_KEY" \
          --batch_size "$bs" \
          --epoch "$EPOCHS" \
          --learning_rate "$lr" \
          --log_dir "$log_dir" \
          --pretrained_ckpt "$ckpt" \
          
        echo -e "[END $(date)] $model | bs=$bs | lr=$lr | ckpt=$ckpt" | tee -a "$LOG_FILE"
        echo "Sleeping for ${SLEEP_DURATION}s..." | tee -a "$LOG_FILE"
        sleep "$SLEEP_DURATION"

      done
    done
  done
done

echo -e "\nAll training runs completed. ($(date))" | tee -a "$LOG_FILE"
