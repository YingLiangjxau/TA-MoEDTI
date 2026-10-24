import numpy as np
import torch
from torch_geometric.nn import to_hetero, GraphConv
from torch.nn import Linear, Dropout, Sigmoid
import random
import torch.nn.functional as F
import torch.nn as nn

import moe
import math


class GNNEncoder(torch.nn.Module):

    def __init__(self, hidden_channels, out_channels, p=0.2):
        super().__init__()

        self.conv_in = GraphConv((-1, -1), hidden_channels, aggr='sum')
        self.conv_med = GraphConv((-1, -1), hidden_channels, aggr='sum')
        self.conv_out = GraphConv((-1, -1), out_channels, aggr='sum')
        self.sigmoid = nn.Sigmoid()
        self.dropout = Dropout(p)

    def forward(self, x, edge_index):
        x = self.conv_in(x, edge_index)
        x = self.sigmoid(x)
        x = self.dropout(x)

        x = self.conv_med(x, edge_index)
        x = self.sigmoid(x)
        x = self.dropout(x)

        x = self.conv_out(x, edge_index)
        x = self.sigmoid(x)
        x = self.dropout(x)
        return x


class EdgeClassifier(torch.nn.Module):

    def __init__(self, hidden_channels):
        super().__init__()
        # 1. 语义投影层
        self.drug_proj = nn.Linear(hidden_channels, hidden_channels)
        self.protein_proj = nn.Linear(hidden_channels, hidden_channels)

        # 2. MoE 交互模块
        self.deep_inter_att = moe.Datt(dim=hidden_channels, nhead=4, dropout=0.2, num_experts=4, top_k=2)

        # 3. 融合 MLP (使用 LayerNorm 替代 BatchNorm)
        fusion_input_dim = hidden_channels * 4
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(hidden_channels * 2),  # ✅ 替换
            nn.Linear(hidden_channels * 2, hidden_channels)
        )

        # 4. 最终分类头 (使用 LayerNorm 替代 BatchNorm)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_channels),  # ✅ 替换
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(hidden_channels),  # ✅ 替换
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, z_dict, hll_dict, edge_label_index):
        row, col = edge_label_index
        device = row.device

        # 1. 获取 GNN 语义并投影
        drug_gnn = self.drug_proj(z_dict['drug'][row]).unsqueeze(1)
        prot_gnn = self.protein_proj(z_dict['protein'][col]).unsqueeze(1)

        # 2. MoE + 交叉注意力处理纯语义交互
        drug_inter, prot_inter = self.deep_inter_att(drug_gnn, prot_gnn)

        # 3. 获取 HLL 拓扑特征
        drug_hll = hll_dict['drug'].to(device)[row]
        prot_hll = hll_dict['protein'].to(device)[col]

        # 4. 深度融合
        combined = torch.cat([
            drug_inter.mean(dim=1),
            prot_inter.mean(dim=1),
            drug_hll,
            prot_hll
        ], dim=-1)

        # 5. 融合 MLP
        fused_feat = self.fusion_mlp(combined)

        # 6. 分类
        out = self.classifier(fused_feat)
        return out.view(-1)


class Model(torch.nn.Module):

    def __init__(self, hidden_channels, data):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = EdgeClassifier(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index, hll_dict):
        z_dict = self.encoder(x_dict, edge_index_dict)
        out = self.decoder(z_dict, hll_dict, edge_label_index)
        return z_dict, out


class EarlyStopper():
    def __init__(self, tolerance=10, min_delta=0.01):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0

    def early_stop(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                return True
        return False


def shuffle_label_data(train_data, a=('drug', 'interaction', 'protein')):
    length = train_data[a].edge_label.shape[0]
    lff = list(np.arange(length))
    random.shuffle(lff)

    train_data[a].edge_label = train_data[a].edge_label[lff]
    train_data[a].edge_label_index = train_data[a].edge_label_index[:, lff]
    return train_data

