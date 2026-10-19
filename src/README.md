# EFI-DL: Deep Learning Models for Forestry (Classification & Regression)

This repository provides a unified deep learning framework for forestry applications, supporting multiple datasets, multiple point cloud models (PointNeXt, PointNet++, DGCNN), and multiple tasks (classification, regression, biomass estimation).

## Create virtual environment
see python_environment_setup.md

## Training a classification model
```
python train_cls.py \
    --model dgcnn \
    --num_category 9 \
    --batch_size 16 \
    --epoch 150 \
    --use_normals \
    --learning_rate 1e-4

```
## Training a regression model
```
python train_cls.py \
    --model dgcnn \
    --num_category 9 \
    --batch_size 16 \
    --epoch 150 \
    --use_normals \
    --learning_rate 1e-4
```

### Code structure

```
src/
  ├── models/     # models
  ├── data/       # processed data
  ├── dataset/    # Dataloader
  ├── utils/      # logger and metrics calculator
  ├── train.py    # Training function
  ├── test.py     # Inference function
  └── config.py   # Hyperparameters
```
## 🙌 Contributing
Pull requests are welcome—feel free to contribute new models, datasets, or improvements.

## 📄 License
MIT License