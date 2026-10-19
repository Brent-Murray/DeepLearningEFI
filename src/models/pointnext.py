import torch
import torch.nn as nn
from pointnext import PointNext, pointnext_s
from .heads import ClassificationHead, RegressionHead

class get_model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # if normals are used: in_dim = 6 (XYZ + normals), otherwise 3
        in_dim = cfg["in_channel"]
        self.num_classes = cfg["num_classes"]
        
        encoder = pointnext_s(in_dim=in_dim)
        self.backbone = PointNext(1024, encoder=encoder)

        self.norm = nn.BatchNorm1d(1024)
        self.act = nn.ReLU()
        
        # 2. Head Selection
        if self.num_classes == 1:
            # Input dimension is 1024 from the backbone
            self.mlp = RegressionHead(input_dim=1024, num_outputs=self.num_classes)
        else:
            # Input dimension is 1024 from the backbone, use dropouts from model config
            self.mlp = ClassificationHead(
                in_dim=1024, 
                num_outputs=self.num_classes, 
                drop1=cfg["drop1"], 
                drop2=cfg["drop2"]
            )
    def forward(self, x):
        xyz = x[:, 0:3, :]
        out = self.norm(self.backbone(x, xyz))
        out = out.mean(dim=-1)
        out = self.act(out)
        
        x = self.mlp(out)
        
        return x, None