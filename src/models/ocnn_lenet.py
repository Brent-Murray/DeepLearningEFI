# Citation: Wang, Peng-Shuai, et al. "O-cnn: Octree-based convolutional neural networks for 3d shape analysis." ACM Transactions On Graphics (TOG) 36.4 (2017): 1-11.

# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import ocnn


from octree_cnn.regressor import Regressor

class OCNN_LeNet(torch.nn.Module):
    r''' Octree-based LeNet for classification.
  '''

    def __init__(self, cfg, in_channels, out_channels, ocnn_stages, ocnn_late_channels, dropout, nempty: bool = False):
        super().__init__()

        self.cfg = cfg
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stages = ocnn_stages
        self.ocnn_late_channels = ocnn_late_channels
        self.dropout = dropout
        self.nempty = nempty

        channels = [in_channels] + [2 ** max(i + 7 - self.stages, 2) for i in range(self.stages)]

        self.convs = torch.nn.ModuleList([ocnn.modules.OctreeConvBnRelu(
            channels[i], channels[i + 1], nempty=nempty) for i in range(self.stages)])
        self.pools = torch.nn.ModuleList([ocnn.nn.OctreeMaxPool(
            nempty) for i in range(self.stages)])
        self.octree2voxel = ocnn.nn.Octree2Voxel(self.nempty)
        self.header = torch.nn.Sequential(
            torch.nn.Dropout(p=self.dropout),  # drop1
            ocnn.modules.FcBnRelu(64 * 64, 128),  # fc1
            torch.nn.Dropout(p=self.dropout),  # drop2
            torch.nn.Linear(128, ocnn_late_channels))  # fc2

        self.regressor = Regressor(in_channels=ocnn_late_channels, num_outputs=self.out_channels)

    def forward(self, batch):
        r''''''

        # Modified from original model
        octree = batch['octree']
        
        # Get model input features from octree object
        data = octree.get_input_feature(self.cfg['ocnn_features'], nempty=self.nempty)

        # Get the depth of the octree
        depth = octree.depth

        for i in range(self.stages):
            d = depth - i
            data = self.convs[i](data, octree, d)
            data = self.pools[i](data, octree, d)
        data = self.octree2voxel(data, octree, depth - self.stages)
        data = self.header(data)

        data = self.regressor(data)

        if self.out_channels == 1:
            data = data.unsqueeze(1)

        return data