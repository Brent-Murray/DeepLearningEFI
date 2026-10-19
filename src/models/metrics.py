# src/utils/metrics.py
import numpy as np
from sklearn.metrics import confusion_matrix, r2_score, mean_squared_error


class MetricsCalculator:
    """Calculates and logs task-specific metrics."""
    def __init__(self, task, num_classes=None):
        self.task = task
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.predictions = []
        self.targets = []
        self.losses = []
        self.n_batches = 0

    def update(self, pred, target, loss_item):
        self.predictions.append(pred.detach().cpu().numpy())
        self.targets.append(target.detach().cpu().numpy())
        self.losses.append(loss_item)
        self.n_batches += 1

    def compute_and_log(self, logger):
        all_pred = np.concatenate(self.predictions, axis=0)
        all_target = np.concatenate(self.targets, axis=0)
        
        avg_loss = np.mean(self.losses) if self.losses else 0.0
        results = {'loss': avg_loss}

        if self.task == 'classification':
            pred_labels = np.argmax(all_pred, axis=1)
            overall_acc = np.mean(pred_labels == all_target)

            # Per-Class Accuracy
            cm = confusion_matrix(all_target, pred_labels)
            class_acc_array = cm.diagonal() / cm.sum(axis=1)
            class_acc = np.mean(class_acc_array[~np.isnan(class_acc_array)])
            
            results.update({'instance_acc': overall_acc, 'class_acc': class_acc, 'confusion_matrix': cm})
            
            logger.info('Validation Loss: %.4f | Instance Acc: %f | Class Acc: %f' % (avg_loss, overall_acc, class_acc))
            
        elif self.task == 'regression':
            pred_values = all_pred.flatten()
            target_values = all_target.flatten()

            r2 = r2_score(target_values, pred_values)
            rmse = np.sqrt(mean_squared_error(target_values, pred_values))
            
            results.update({'r2': r2, 'rmse': rmse})
            
            logger.info('Validation Loss: %.4f | R2 Score: %f | RMSE: %f' % (avg_loss, r2, rmse))

        return results
