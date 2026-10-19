#!/bin/bash

SLEEP_DURATION=10  # seconds
EPOCHS=100

LOG_FILE="bash_reg_log.txt"
exec > >(tee -a bash_reg_log.txt) 2>&1

DATASET_KEY="petawawa_reg"

echo "Starting training... ($(date))" | tee -a "$LOG_FILE"

models=(
  dgcnn
)

batch_sizes=(8 16)
learning_rates=(0.0001 0.001)

for model in "${models[@]}"; do
  for bs in "${batch_sizes[@]}"; do
    for lr in "${learning_rates[@]}"; do

      # format lr tag
      if [[ "$lr" == "0.0001" ]]; then
        lr_tag="e4"
      elif [[ "$lr" == "0.001" ]]; then
        lr_tag="e3"
      fi

      log_dir="${DATASET_KEY}_${model}_bs${bs}_lr${lr_tag}"

      echo -e "\n[START $(date)] $model | bs=$bs | lr=$lr" | tee -a "$LOG_FILE"

      python train_pl.py \
        --model "$model" \
        --mode "train" \
        --dataset_config_key "$DATASET_KEY" \
        --batch_size "$bs" \
        --epoch "$EPOCHS" \
        --learning_rate "$lr" \
        --log_dir "$log_dir" 

      echo -e "[END $(date)] $model | bs=$bs | lr=$lr" | tee -a "$LOG_FILE"
      echo "Sleeping for ${SLEEP_DURATION}s..." | tee -a "$LOG_FILE"
      sleep "$SLEEP_DURATION"

    done
  done
done

echo -e "\nAll training runs completed. ($(date))" | tee -a "$LOG_FILE"
