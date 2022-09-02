# -*- coding: utf-8 -*-
# @Time         : 2022/7/23 13:30
# @Author       : Yufan Liu
# @Description  : Model

import torch_geometric.nn as pyg_nn
import torch.nn as nn
import torch
from copy import deepcopy


# Model 1
class GCNModel(nn.Module):
    """
    Base model, consisting of GAT, GCN combination.
    """

    def __init__(self):
        super(GCNModel, self).__init__()
        self.conv1 = pyg_nn.GATConv(1024, 32, heads=8)
        self.conv2 = pyg_nn.GCNConv(256, 128)
        self.conv3 = pyg_nn.GATConv(128, 64, heads=8, concat=False)
        self.conv4 = pyg_nn.GCNConv(64, 2)
        # self.conv4 = pyg_nn.GCNConv(64, 32)
        # self.conv5 = pyg_nn.GCNConv(32, 2)

        self.norm0 = nn.LayerNorm(256)
        self.norm1 = nn.LayerNorm(128)
        self.norm2 = nn.LayerNorm(64)

        self.mlp1 = nn.Linear(1024, 256)
        self.mlp2 = nn.Linear(256, 128)
        self.mlp3 = nn.Linear(128, 64)
        self.mlp4 = nn.Linear(64, 2)
        # self.mlp5 = nn.Linear(32, 2)

        self.act = nn.ReLU()

        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index):
        x = self.dropout(x)
        x = self.conv1(x, edge_index) + self.mlp1(x)
        x = self.norm0(x)
        x = self.act(x)

        x = self.dropout(x)
        x = self.conv2(x, edge_index) + self.mlp2(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.dropout(x)
        x = self.conv3(x, edge_index) + self.mlp3(x)
        x = self.norm2(x)
        x = self.act(x)

        x = self.dropout(x)
        x = self.conv4(x, edge_index) + self.mlp4(x)

        return x

# model 2
class GTM(nn.Module):
    """
    simple Transformer for graph nodes
    """

    def __init__(self):
        super(GTM, self).__init__()
        encoder1 = nn.TransformerEncoderLayer(d_model=1024, nhead=4)
        encoder2 = nn.TransformerEncoderLayer(d_model=64, nhead=4)
        self.transformer_encoder1 = nn.TransformerEncoder(encoder1, num_layers=2)
        self.transformer_encoder2 = nn.TransformerEncoder(encoder2, num_layers=2)

        self.mlp1 = nn.Linear(1024, 64)
        self.mlp2 = nn.Linear(64, 2)

        self.input_block = nn.Sequential(nn.LayerNorm(1024),
                                         nn.Linear(1024, 64),
                                         nn.LeakyReLU())

        self.mlp = nn.Linear(64, 2)

    def forward(self, x):
        x = self.input_block(x)  # [batch, 64]
        # x = self.transformer_encoder1(x)
        x = self.transformer_encoder2(x.unsqueeze(0))
        return x.squeeze(0)


# model 1 + 2 combinations
class GraphCombine(nn.Module):
    """
    Concat GT and MPNN
    """
    def __init__(self):
        super(GraphCombine, self).__init__()
        self.mpnn = GCNModel()
        self.gt = GTM()
        self.head = nn.Linear(64 + 64, 2)

    def forward(self, x ,edge_index):
        y = deepcopy(x)
        x = self.mpnn(x, edge_index)
        y = self.gt(y)

        return self.head(torch.concat((x, y), axis=1))

