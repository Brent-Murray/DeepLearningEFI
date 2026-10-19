import torch.nn as nn
import torch.nn.functional as F
from .pointnet2_utils import PointNetSetAbstraction
from .heads import ClassificationHead, RegressionHead

class get_model(nn.Module):
    def __init__(self, cfg):
        super(get_model, self).__init__()
        in_channel = cfg["in_channel"]
        self.normal_channel = cfg["use_normal"]
        
        self.num_classes = cfg["num_classes"]
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        
        if self.num_classes == 1:
            self.mlp = RegressionHead(1024, self.num_classes)
        else:
            self.mlp = ClassificationHead(
                in_dim=1024, 
                num_outputs=self.num_classes, 
                drop1=cfg["drop1"], 
                drop2=cfg["drop2"]
            )

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        
        x = self.mlp(x)

        return x, l3_points

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss