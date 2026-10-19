# models/loss_utils.py
import torch.nn as nn
import torch.nn.functional as F


def get_loss_function(task):
    """
    Returns the core loss function (either nn.Module or functional).
    """
    if task == 'classification':
        # Using CrossEntropyLoss is simplest if input is raw logits.
        return F.nll_loss # nn.CrossEntropyLoss()
    elif task == 'regression':
        return F.smooth_l1_loss
    else:
        raise ValueError(f"Unknown task type: {task}")

def calculate_total_loss(pred, target, trans_feat, task, mat_diff_loss_scale=0.001):
    """
    Calculates the loss including task-specific loss and regularization.
    """
    # 1. Get Core Task Loss
    core_criterion = get_loss_function(task)
    
    if task == 'regression':
        # Target must be float for regression. Pred [B, 1] -> [B].
        target = target.float()

        if pred.shape != target.shape:
            pred = pred.view(-1)
        
        assert pred.shape == target.shape, f"Pred shape {pred.shape} and target shape {target.shape} do not match."

        core_loss = core_criterion(pred, target)
    else:
        # Classification: F.nll_loss REQUIRES log-probabilities (log_softmax).
        core_loss = core_criterion(pred, target.long())

    # 2. Add Feature Transform Regularization (PointNet specific)
    reg_loss = 0.0
    if trans_feat is not None and mat_diff_loss_scale > 0:
        from .pointnet_utils import feature_transform_reguliarzer
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        reg_loss = mat_diff_loss * mat_diff_loss_scale

    return core_loss + reg_loss