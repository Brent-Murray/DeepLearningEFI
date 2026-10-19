# models/heads.py
import torch.nn as nn
import torch.nn.functional as F

class ClassificationHead(nn.Module):
    def __init__(self, in_dim, num_outputs, drop1=0.5, drop2=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(drop1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(drop2),
            nn.Linear(256, num_outputs),
        )

    def forward(self, x):
        return F.log_softmax(self.net(x), dim=-1)   # logits [B, num_classes]


class RegressionHead(nn.Module):
    def __init__(self, input_dim, num_outputs):
        super().__init__()
        self.num_outputs = num_outputs

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_outputs)
        )

    def forward(self, feat):
        
        return self.net(feat)
