import torch.nn as nn
from .pointnet_utils import PointNetEncoder
from .heads import ClassificationHead, RegressionHead

class get_model(nn.Module):
    def __init__(self, cfg):
        super(get_model, self).__init__()
        channel = cfg["in_channel"]
        self.num_classes = cfg["num_classes"]
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        if self.num_classes == 1:
            self.mlp = RegressionHead(1024, self.num_classes)
        else:
            self.mlp = ClassificationHead(
                in_dim=1024, 
                num_outputs=self.num_classes, 
                drop1=cfg["drop1"], 
                drop2=cfg["drop2"]
            )
    def forward(self, x):
        # x shape: [B, C, N]
        x, _, trans_feat = self.feat(x)
        x = self.mlp(x)
        return x, trans_feat