import pytorch_lightning as pl
import torch
import importlib
import numpy as np
from sklearn.metrics import confusion_matrix, r2_score, mean_squared_error

from .loss_utils import calculate_total_loss 
from .metrics import MetricsCalculator

class EfiModelModule(pl.LightningModule):
    def __init__(self, cfg, args):
        super().__init__()
        # Store hyperparameters for W&B/checkpointing
        self.save_hyperparameters(cfg, args) 
        self.cfg = cfg
        self.args = args
        self.task = cfg['task']
        self.batch_size = args.batch_size

        # 1. Load Model Backbone
        # NOTE: This import assumes a specific package structure (e.g., models.model_name)
        model_module = importlib.import_module(f"models.{cfg['model_name']}")
        self.model = model_module.get_model(cfg)

        # Load pretrained model checkpoint
        if args.pretrained_ckpt is not None:
            self.load_checkpoint(args.pretrained_ckpt)

        # 2. Setup Metric Calculators (one for validation, one for testing)
        self.val_metrics = MetricsCalculator(self.task, cfg['num_classes'])
        self.test_metrics = MetricsCalculator(self.task, cfg['num_classes'])


    def forward(self, points):
        """Standard model forward pass."""
        return self.model(points)

    def _shared_step(self, batch):
        """Shared logic for training, validation, and test step."""
        
        if self.cfg['model_name'] == 'ocnn':
            
            target = batch['target']

            # 1. Forward Pass
            pred, trans_feat = self(batch)
        else:
            points, target, _ = batch

            # Transpose [B, N, C] -> [B, C, N] and enforce float32
            points = points.transpose(2, 1).float() 
            
            # 1. Forward Pass
            pred, trans_feat = self(points)
        
        
        # 2. Calculate Loss
        loss = calculate_total_loss(
            pred, 
            target, 
            trans_feat, 
            task=self.task,
            mat_diff_loss_scale=self.cfg.get('mat_diff_loss_scale', 0.0)
        )
        
        return loss, pred, target

    # --- Training ---
    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch)
        
        # Log basic training loss and learning rate for W&B
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('train/lr', self.optimizers().param_groups[0]['lr'], on_step=True, on_epoch=False, batch_size=self.batch_size)

        return loss

    # --- Validation ---
    def validation_step(self, batch, batch_idx):
        loss, pred, target = self._shared_step(batch)
        
        # 1. Log validation loss (PL automatically averages this over the epoch)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        
        # 2. Update the full MetricsCalculator for end-of-epoch aggregation
        self.val_metrics.update(pred, target, 0.0)
        
        return loss

    def on_validation_epoch_end(self):
        """
        Calculates and logs all final validation metrics using W&B.
        """
        # Compute metrics from the aggregated predictions/targets
        all_pred = np.concatenate(self.val_metrics.predictions, axis=0)
        all_target = np.concatenate(self.val_metrics.targets, axis=0)
        
        val_logs = {}

        if self.task == 'classification':
            pred_labels = np.argmax(all_pred, axis=1)
            overall_acc = np.mean(pred_labels == all_target)

            # Per-Class Accuracy
            cm = confusion_matrix(all_target, pred_labels)
            # Handle division by zero warning if a class has no samples in the batch
            with np.errstate(divide='ignore', invalid='ignore'):
                 class_acc_array = cm.diagonal() / cm.sum(axis=1)
            class_acc = np.mean(class_acc_array[~np.isnan(class_acc_array)])
            
            # Log all classification metrics to W&B
            val_logs['val/overall_acc'] = overall_acc
            val_logs['val/class_acc'] = class_acc
            
        elif self.task == 'regression':
            pred_values = all_pred.flatten()
            target_values = all_target.flatten()
            
            # Compute R2 and RMSE
            r2 = r2_score(target_values, pred_values)
            rmse = np.sqrt(mean_squared_error(target_values, pred_values))
            
            # Log all regression metrics to W&B
            val_logs['val/r2_score'] = r2
            val_logs['val/rmse'] = rmse

        # PL logging handles synchronization across GPUs
        self.log_dict(val_logs, on_step=False, on_epoch=True, prog_bar=False, batch_size=self.batch_size)
        
        # Reset metric calculator for the next epoch
        self.val_metrics.reset()

    # --- Test Step ---
    def test_step(self, batch, batch_idx):
        """
        Dedicated step for final, unbiased evaluation on the test set.
        Runs when trainer.test() is called.
        """
        loss, pred, target = self._shared_step(batch)
        
        # 1. Log test loss
        self.log('test/loss', loss, on_step=False, on_epoch=True, batch_size=self.batch_size)
        
        # 2. Update the full MetricsCalculator for end-of-epoch aggregation
        self.test_metrics.update(pred, target, 0.0)
        
        return loss

    def on_test_epoch_end(self):
        """
        Calculates and logs all final test metrics, mirroring the validation logic.
        """
        # Compute metrics from the aggregated predictions/targets
        all_pred = np.concatenate(self.test_metrics.predictions, axis=0)
        all_target = np.concatenate(self.test_metrics.targets, axis=0)
        
        test_logs = {}

        if self.task == 'classification':
            pred_labels = np.argmax(all_pred, axis=1)
            overall_acc = np.mean(pred_labels == all_target)

            # Per-Class Accuracy
            cm = confusion_matrix(all_target, pred_labels)
            with np.errstate(divide='ignore', invalid='ignore'):
                 class_acc_array = cm.diagonal() / cm.sum(axis=1)
            class_acc = np.mean(class_acc_array[~np.isnan(class_acc_array)])
            
            # Log all classification metrics
            test_logs['test/overall_acc'] = overall_acc
            test_logs['test/class_acc'] = class_acc
            
        elif self.task == 'regression':
            pred_values = all_pred.flatten()
            target_values = all_target.flatten()
            
            # Compute R2 and RMSE
            r2 = r2_score(target_values, pred_values)
            rmse = np.sqrt(mean_squared_error(target_values, pred_values))
            
            # Log all regression metrics
            test_logs['test/r2_score'] = r2
            test_logs['test/rmse'] = rmse

        # PL logging handles synchronization across GPUs
        self.log_dict(test_logs, on_step=False, on_epoch=True, prog_bar=False, batch_size=self.batch_size)
        
        # Reset metric calculator for the next run
        self.test_metrics.reset()


    # --- Prediction / Inference ---
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Used for generating raw output/predictions on unseen data.
        Does not calculate loss or metrics.
        Runs when trainer.predict() is called.
        """

        if self.cfg['model_name'] == 'ocnn':
            
            target = batch['target']
            pid = batch['pid']
            pred, _ = self(batch)
        
        else:

            points, target, pid = batch 
            
            # Transpose [B, N, C] -> [B, C, N] and enforce float32
            points = points.transpose(2, 1).float() 
            
            # 1. Forward Pass
            # Assuming self(points) returns (pred, trans_feat), we only return the prediction (pred)
            pred, _ = self(points)
            if self.task == 'classification':
                pred = torch.argmax(pred, dim=1)
        
        # Return only the raw prediction tensor for the user
        return {
            "plot_id": pid,
            "pred": pred.detach().cpu(),
            "gt": target.detach().cpu()
        }

    # --- Optimizer Configuration ---
    def configure_optimizers(self):
        """Defines the optimizer and scheduler as required by PL."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.args.decay_rate
        )
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
        
        # PL requires the scheduler to be returned in a dictionary format
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler, 
                'interval': 'epoch', # Run scheduler step after each epoch
            }
        }
    
    def load_checkpoint(self, path: str) -> None:
        """
        NOTE: This method is currently only set up to load OCNN model checkpoints.
        """
        
        #Load the pytorch lightning checkpoint dict, only keeping the state_dict item
        checkpoint = torch.load(path, weights_only=True)['state_dict']

        # Renam 'model.' subsrting to 'model.ocnn.' to match current model structure
        checkpoint = {k.replace("model.", "model.ocnn."): v for k, v in checkpoint.items()}

        #Drop regressor module (different number of outputs leads to mismatch)
        checkpoint = {k: v for k, v in checkpoint.items() if "regressor" not in k}

        #Load the state dict weights and biases to OCNN
        missing_keys, unexpected_keys = self.load_state_dict(checkpoint, strict=False)  # type: ignore

        # Remove missing MLP keys since those are not present in backbone
        missing_keys = [k for k in missing_keys if "mlp" not in k]

        #Check that only missing keys were from the regressor module
        missing_keys = [k for k in missing_keys if "regressor" not in k]

        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")

        print(f"\033[92mLoaded pretrained weights from: {path}.\033[0m")