import torch.nn as nn
from .pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction
from .heads import ClassificationHead, RegressionHead

class get_model(nn.Module):
    def __init__(self, cfg):
        super(get_model, self).__init__()
        self.normal_channel = cfg["use_normal"] 
        in_channel = 3 if self.normal_channel else 0
        self.num_classes = cfg["num_classes"]
        
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        
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