# -*- coding: utf-8 -*-
# @Time         : 2022/7/23 13:30
# @Author       : Yufan Liu
# @Description  : Model and train

import torch_geometric.nn as pyg_nn
import torch.nn as nn
import torch.nn.functional as F


class GCNModel(nn.Module):
    def __init__(self):
        super(GCNModel, self).__init__()
        self.conv1 = pyg_nn.GATConv(1024, 32, heads=8)
        self.conv2 = pyg_nn.GCNConv(256, 128)
        self.conv3 = pyg_nn.GATConv(128, 64, heads=8, concat=False)
        self.conv4 = pyg_nn.GCNConv(64, 2)
        #self.conv4 = pyg_nn.GCNConv(64, 32)
        #self.conv5 = pyg_nn.GCNConv(32, 2)

        self.norm0 = nn.LayerNorm(256)
        self.norm1 = nn.LayerNorm(128)
        self.norm2 = nn.LayerNorm(64)

        self.mlp1 = nn.Linear(1024, 256)
        self.mlp2 = nn.Linear(256, 128)
        self.mlp3 = nn.Linear(128, 64)
        self.mlp4 = nn.Linear(64, 2)
        #self.mlp5 = nn.Linear(32, 2)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index):
        # x = nn.dropout(x, 0.3)
        x = self.conv1(x, edge_index) + self.mlp1(x)
        x = self.norm0(x)
        #x = self.dropout(x)

        x = self.conv2(x, edge_index) + self.mlp2(x)
        x = self.norm1(x)
        # x = self.dropout(x)

        x = self.conv3(x, edge_index) + self.mlp3(x)
        x = self.norm2(x)
        #x = self.dropout(x)

        x = self.conv4(x, edge_index) + self.mlp4(x)
        #x = F.dropout(x, 0.5)
        # x = self.conv5(x, edge_index) + self.mlp5(x)
        # x = F.dropout(x, 0.5)
        return x

