#!/usr/bin/env python3
"""
SimpleGKAN 
Two KAN-based graph layers + KAN head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool

from kan_layers_v25 import KANLinear


class GKANLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, grid_size=8, spline_order=3, dropout=0.3):
        super().__init__(aggr='add')
        self.node_kan = KANLinear(in_channels, out_channels, grid_size=grid_size, spline_order=spline_order)
        self.edge_kan = KANLinear(1, out_channels, grid_size=grid_size, spline_order=spline_order)
        self.msg_kan  = KANLinear(out_channels, out_channels, grid_size=grid_size, spline_order=spline_order)

        self.norm = nn.LayerNorm(out_channels)
        self.drop = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is None:
            raise RuntimeError("GKANLayer expects edge_attr [E] or [E,1].")
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        x_t = self.node_kan(x)               # [N, C]
        e_t = self.edge_kan(edge_attr)       # [E, C]
        out = self.propagate(edge_index, x=x_t, e=e_t)
        out = self.norm(out)
        out = self.drop(self.act(out))
        return out

    def message(self, x_j, e):
        # fuse sender x_j with transformed edge e via msg_kan
        return self.msg_kan(x_j + e)


class SimpleGKAN(nn.Module):
    """
    Two GKAN layers (Cin=5 -> 32 -> 16) + global mean pool + KAN head -> 2 classes.
    """
    def __init__(self, input_dim=5, hidden_dim=32, num_classes=2, grid_size=8, spline_order=3, dropout=0.3, residual=True):
        super().__init__()
        self.residual = residual

        self.g1 = GKANLayer(input_dim, hidden_dim, grid_size, spline_order, dropout)
        self.g2 = GKANLayer(hidden_dim, 16, grid_size, spline_order, dropout)

        self.res1 = nn.Identity() if input_dim == hidden_dim else nn.Linear(input_dim, hidden_dim, bias=False)
        self.res2 = nn.Identity() if hidden_dim == 16 else nn.Linear(hidden_dim, 16, bias=False)

        # Network-wise pooling and attention
        self.yeo_networks = {
            1: list(range(43, 54)),   # Visual
            2: list(range(1, 21)) + list(range(57, 71)),  # Somatomotor
            3: list(range(59, 68)),   # Dorsal Attention
            4: list(range(29, 31)) + list(range(61, 67)),  # Ventral Attention
            5: list(range(35, 42)) + list(range(83, 90)),  # Limbic
            6: list(range(3, 16)) + list(range(59, 68)),   # Frontoparietal
            7: list(range(23, 28)) + list(range(35, 40)) + list(range(65, 68))  # Default Mode
        }
        self.network_attention = nn.ModuleDict({
            str(net_id): nn.Sequential(
                nn.Linear(16, 8),
                nn.Tanh(),
                nn.Linear(8, 1)
            ) for net_id in self.yeo_networks.keys()
        })

        self.post_pool_drop = nn.Dropout(dropout)
        self.head1 = KANLinear(16 * 7, 32, grid_size=grid_size, spline_order=spline_order)  # 7 networks * 16 features
        self.norm = nn.LayerNorm(32)
        self.head2 = KANLinear(32, num_classes, grid_size=grid_size, spline_order=spline_order)
        self.act = nn.SiLU()

    def forward(self, batch):
        x, ei, ea, b = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        if ea is None:
            raise RuntimeError("SimpleGKAN expects edge_attr.")

        h1 = self.g1(x, ei, ea)
        x = self.res1(x) + h1 if self.residual else h1
        h2 = self.g2(x, ei, ea)
        x = self.res2(x) + h2 if self.residual else h2

        x = self.network_wise_pool(x, b)
        x = self.post_pool_drop(x)
        x = self.head1(x)
        x = self.norm(self.act(x))
        x = self.head2(x)
        return x

    def network_wise_pool(self, x, batch):
        """Network-wise pooling with learned attention weights"""
        device = x.device
        network_embeddings = []

        for net_id, node_indices in self.yeo_networks.items():
            # Filter valid indices (0-115 for 116 regions)
            valid_indices = [i for i in node_indices if i < x.size(0)]

            if not valid_indices:
                # If no valid indices, use zero embedding
                network_embeddings.append(torch.zeros(1, 16, device=device))
                continue

            if batch is not None:
                # Handle batched data - this is more complex, implement simple version first
                network_nodes = x[valid_indices]  # [num_nodes_in_network, features]

                # Compute attention weights
                attn_scores = self.network_attention[str(net_id)](network_nodes)  # [num_nodes_in_network, 1]
                attn_weights = F.softmax(attn_scores, dim=0)

                # Weighted average
                network_emb = torch.sum(network_nodes * attn_weights, dim=0, keepdim=True)  # [1, features]
            else:
                # Single sample case
                network_nodes = x[valid_indices]
                attn_scores = self.network_attention[str(net_id)](network_nodes)
                attn_weights = F.softmax(attn_scores, dim=0)
                network_emb = torch.sum(network_nodes * attn_weights, dim=0, keepdim=True)

            network_embeddings.append(network_emb)

        # Concatenate all network embeddings
        return torch.cat(network_embeddings, dim=1)  # [batch_size, 16*7]

    def compute_spline_regularization(self):
        reg = 0.0
        for m in [self.g1.node_kan, self.g1.edge_kan, self.g1.msg_kan,
                  self.g2.node_kan, self.g2.edge_kan, self.g2.msg_kan,
                  self.head1, self.head2]:
            if hasattr(m, "spline_weight"):
                reg = reg + m.spline_weight.pow(2).mean()
        return reg
