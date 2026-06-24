#!/usr/bin/env python3
"""
Baseline GNN models (parameter-matched to the GKAN model).
GCNBaseline: 3 GCN layers (160, 160, 80) with batch norm, dropout, linear head.
GATBaseline: 3 GAT layers (hidden_channels=64, heads=4) with batch norm, dropout, linear head.
MLPBaseline: included for completeness (not parameter-matched by default).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

# -------------------------------------------------------
# Utility
# -------------------------------------------------------
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -------------------------------------------------------
# MLP (optional, not param-matched)
# -------------------------------------------------------
class MLPBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 128), num_classes=2, dropout=0.5):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i+1]),
                nn.BatchNorm1d(dims[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], num_classes)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight);
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)


# -------------------------------------------------------
# GCN (param target ~ 40k)
#   signature: GCNBaseline(input_dim, hidden_dims=(160,160,80), num_classes=2, dropout=0.5)
# -------------------------------------------------------
class GCNBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dims=(160, 160, 80), num_classes=2, dropout=0.5, pooling_mode="global"):
        super().__init__()
        assert len(hidden_dims) == 3, "hidden_dims must be a 3-tuple, e.g., (160,160,80)"
        h1, h2, h3 = hidden_dims

        self.conv1 = GCNConv(input_dim, h1)
        self.bn1   = nn.BatchNorm1d(h1)
        self.do1   = nn.Dropout(dropout)

        self.conv2 = GCNConv(h1, h2)
        self.bn2   = nn.BatchNorm1d(h2)
        self.do2   = nn.Dropout(dropout)

        self.conv3 = GCNConv(h2, h3)
        self.bn3   = nn.BatchNorm1d(h3)
        self.do3   = nn.Dropout(dropout)

        self.pooling_mode = pooling_mode
        self.num_networks = 7

        # Calculate classifier input dimension
        if pooling_mode == "network":
            classifier_input_dim = h3 * self.num_networks
        else:
            classifier_input_dim = h3

        self.head = nn.Linear(classifier_input_dim, num_classes)
        self._init()

    def _init(self):
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None: nn.init.zeros_(self.head.bias)

    def forward(self, data):
        x, ei, b = data.x, data.edge_index, getattr(data, 'batch', None)

        x = self.conv1(x, ei); x = self.bn1(x); x = F.relu(x); x = self.do1(x)
        x = self.conv2(x, ei); x = self.bn2(x); x = F.relu(x); x = self.do2(x)
        x = self.conv3(x, ei); x = self.bn3(x); x = F.relu(x); x = self.do3(x)

        # Apply pooling based on mode
        if self.pooling_mode == "network":
            x = self._network_pooling(x, data)
        else:
            x = global_mean_pool(x, b) if b is not None else x.mean(dim=0, keepdim=True)

        return self.head(x)

    def _network_pooling(self, x, data):
        """
        Pools node features per (graph, network) and flattens to [B, num_networks * F].
        Expects:
        - data.batch: [N] in [0..B-1]
        - data.net_ids: [N] in [0..num_networks-1]
        """
        if not hasattr(data, "batch"):
            raise RuntimeError("network_pooling requires data.batch")
        if not hasattr(data, "net_ids"):
            # Fallback to global pooling if net_ids missing
            from torch_geometric.nn import global_mean_pool
            return global_mean_pool(x, data.batch) if data.batch is not None else x.mean(dim=0, keepdim=True)

        device = x.device
        b = data.batch.to(device=device, dtype=torch.long)
        net_ids = data.net_ids.to(device=device, dtype=torch.long)

        if b.numel() != x.size(0) or net_ids.numel() != x.size(0):
            raise RuntimeError(f"Size mismatch: x={x.size(0)}, batch={b.numel()}, net_ids={net_ids.numel()}")

        B = int(b.max().item()) + 1 if b.numel() > 0 else 1
        num_nets = int(getattr(self, "num_networks", 7))

        # Clamp any out-of-range net ids (just in case)
        if net_ids.min() < 0 or net_ids.max() >= num_nets:
            net_ids = torch.clamp(net_ids, 0, num_nets - 1)

        combined_idx = b * num_nets + net_ids
        dim_size = B * num_nets

        # --- HARD GUARDS (with CPU fallback for clear error message) ---
        bad_neg = (combined_idx < 0).nonzero(as_tuple=False).view(-1)
        bad_big = (combined_idx >= dim_size).nonzero(as_tuple=False).view(-1)
        if bad_neg.numel() or bad_big.numel():
            # move to CPU for clearer print, then raise
            ci_cpu = combined_idx.detach().cpu()
            samples = torch.cat([bad_neg[:5], bad_big[:5]]).unique().tolist() if (bad_neg.numel() or bad_big.numel()) else []
            print(f"[network_pooling] B={B}, num_nets={num_nets}, dim_size={dim_size}")
            if bad_neg.numel():
                idxs = bad_neg[:10].tolist()
                print(f"[network_pooling] Negative combined_idx at positions {idxs}, e.g. values: {[int(ci_cpu[i]) for i in idxs]}")
            if bad_big.numel():
                idxs = bad_big[:10].tolist()
                print(f"[network_pooling] OOB combined_idx >= {dim_size} at positions {idxs}, e.g. values: {[int(ci_cpu[i]) for i in idxs]}")
            raise RuntimeError("combined_idx out of bounds")

        # Ensure correct dtype/device
        if combined_idx.dtype != torch.long:
            combined_idx = combined_idx.long()
        if combined_idx.device != x.device:
            combined_idx = combined_idx.to(x.device)

        # --- IMPORTANT: pass dim_size! ---
        pooled = global_mean_pool(x, combined_idx, size=dim_size)  # [B*num_nets, F]

        F = x.size(1)
        pooled = pooled.view(B, num_nets, F).reshape(B, num_nets * F)
        return pooled


# -------------------------------------------------------
# GAT (parameter target ~ 40-42k)
#   signature: GATBaseline(input_dim, num_classes=2, hidden_channels=64, heads=4, num_layers=3, dropout=0.5)
# -------------------------------------------------------
class GATBaseline(nn.Module):
    def __init__(self, input_dim, num_classes=2, hidden_channels=64, heads=4, num_layers=3, dropout=0.5, pooling_mode="global"):
        super().__init__()
        assert num_layers >= 2, "Use at least 2 GAT layers."

        self.layers = nn.ModuleList()
        self.bns    = nn.ModuleList()
        self.dos    = nn.ModuleList()

        # Layer 1
        self.layers.append(GATConv(in_channels=input_dim, out_channels=hidden_channels,
                                   heads=heads, concat=True, dropout=dropout))
        self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
        self.dos.append(nn.Dropout(dropout))
        C = hidden_channels * heads  # current feature size

        # Middle layers (keep concat=True)
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(in_channels=C, out_channels=hidden_channels,
                                       heads=heads, concat=True, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
            self.dos.append(nn.Dropout(dropout))
            C = hidden_channels * heads

        # Final layer (concat=False to keep feature size manageable)
        self.layers.append(GATConv(in_channels=C, out_channels=hidden_channels,
                                   heads=1, concat=False, dropout=dropout))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.dos.append(nn.Dropout(dropout))
        C = hidden_channels  # final node feature size

        self.pooling_mode = pooling_mode
        self.num_networks = 7

        # Calculate classifier input dimension
        if pooling_mode == "network":
            classifier_input_dim = C * self.num_networks
        else:
            classifier_input_dim = C

        self.head = nn.Linear(classifier_input_dim, num_classes)
        self._init()

    def _init(self):
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None: nn.init.zeros_(self.head.bias)

    def forward(self, data):
        x, ei, b = data.x, data.edge_index, getattr(data, 'batch', None)

        for i, (gat, bn, do) in enumerate(zip(self.layers, self.bns, self.dos)):
            x = gat(x, ei)
            x = bn(x)
            if i < len(self.layers) - 1:
                x = F.elu(x)
            x = do(x)

        # Apply pooling based on mode
        if self.pooling_mode == "network":
            x = self._network_pooling(x, data)
        else:
            x = global_mean_pool(x, b) if b is not None else x.mean(dim=0, keepdim=True)

        return self.head(x)

    def _network_pooling(self, x, data):
        """
        Pools node features per (graph, network) and flattens to [B, num_networks * F].
        Expects:
        - data.batch: [N] in [0..B-1]
        - data.net_ids: [N] in [0..num_networks-1] (0-BASED, NOT 1-BASED!)
        """
        if not hasattr(data, "batch"):
            raise RuntimeError("network_pooling requires data.batch")
        if not hasattr(data, "net_ids"):
            # Fallback to global pooling if net_ids missing
            from torch_geometric.nn import global_mean_pool
            return global_mean_pool(x, data.batch) if data.batch is not None else x.mean(dim=0, keepdim=True)

        device = x.device
        b = data.batch.to(device=device, dtype=torch.long)
        net_ids = data.net_ids.to(device=device, dtype=torch.long)

        if b.numel() != x.size(0) or net_ids.numel() != x.size(0):
            raise RuntimeError(f"Size mismatch: x={x.size(0)}, batch={b.numel()}, net_ids={net_ids.numel()}")

        B = int(b.max().item()) + 1 if b.numel() > 0 else 1
        num_nets = int(getattr(self, "num_networks", 7))

        # Clamp any out-of-range net ids (just in case)
        if net_ids.min() < 0 or net_ids.max() >= num_nets:
            net_ids = torch.clamp(net_ids, 0, num_nets - 1)

        # net_ids are already 0-based (0-6)
        combined_idx = b * num_nets + net_ids
        dim_size = B * num_nets

        # HARD GUARDS
        bad_neg = (combined_idx < 0).nonzero(as_tuple=False).view(-1)
        bad_big = (combined_idx >= dim_size).nonzero(as_tuple=False).view(-1)
        if bad_neg.numel() or bad_big.numel():
            # move to CPU for clearer print, then raise
            ci_cpu = combined_idx.detach().cpu()
            samples = torch.cat([bad_neg[:5], bad_big[:5]]).unique().tolist() if (bad_neg.numel() or bad_big.numel()) else []
            print(f"[network_pooling] B={B}, num_nets={num_nets}, dim_size={dim_size}")
            if bad_neg.numel():
                idxs = bad_neg[:10].tolist()
                print(f"[network_pooling] Negative combined_idx at positions {idxs}, e.g. values: {[int(ci_cpu[i]) for i in idxs]}")
            if bad_big.numel():
                idxs = bad_big[:10].tolist()
                print(f"[network_pooling] OOB combined_idx >= {dim_size} at positions {idxs}, e.g. values: {[int(ci_cpu[i]) for i in idxs]}")
            raise RuntimeError("combined_idx out of bounds")

        # Ensure correct dtype/device
        if combined_idx.dtype != torch.long:
            combined_idx = combined_idx.long()
        if combined_idx.device != x.device:
            combined_idx = combined_idx.to(x.device)

        # --- IMPORTANT: pass dim_size! ---
        pooled = global_mean_pool(x, combined_idx, size=dim_size)  # [B*num_nets, F]

        F = x.size(1)
        pooled = pooled.view(B, num_nets, F).reshape(B, num_nets * F)
        return pooled
