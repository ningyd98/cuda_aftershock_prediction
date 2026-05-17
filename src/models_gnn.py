from __future__ import annotations

import math

import torch
from torch import nn


class SpatialGraphConv(nn.Module):
    """
    空间图卷积层：基于事件间距离的高斯核邻接矩阵。

    A_ij = exp(-d_ij² / (2σ²)), normsed row-wise.
    """

    def __init__(self, in_dim: int, out_dim: int, sigma: float = 50.0) -> None:
        super().__init__()
        self.sigma = sigma
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D_in) node features
            coords: (B, N, 2) node spatial coordinates (rel_x_km, rel_y_km)
            mask: (B, N) True = padding
        Returns:
            (B, N, D_out) updated node features
        """
        B, N, _ = x.shape
        # Pairwise distance matrix
        coord_diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # (B, N, N, 2)
        dists = torch.norm(coord_diff, dim=-1)  # (B, N, N)
        # Gaussian kernel adjacency
        adj = torch.exp(-(dists ** 2) / (2 * self.sigma ** 2))  # (B, N, N)
        # Mask invalid nodes
        valid_mask = (~mask).float().unsqueeze(-1)  # (B, N, 1)
        adj = adj * valid_mask * valid_mask.transpose(1, 2)
        # Row normalization
        deg = adj.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        adj_norm = adj / deg
        # Message passing
        msg = self.linear(x)  # (B, N, D_out)
        out = torch.bmm(adj_norm, msg)  # (B, N, D_out)
        return out


class TemporalGRU(nn.Module):
    """时序 GRU 编码层。"""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2) -> None:
        super().__init__()
        self.gru = nn.GRU(
            in_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
        )
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) event features
            mask: (B, N) True = padding
        Returns:
            (B, N, D_hidden) temporal encoding per node
        """
        # Pack padded sequence
        lengths = (~mask).sum(dim=1).cpu()  # (B,)
        # Since nodes within a sequence have the same temporal aspect,
        # we process per-batch-item separately (bottleneck but simple)
        B, N, D = x.shape
        outputs = torch.zeros(B, N, self.proj.out_features, device=x.device)
        for b in range(B):
            valid_len = int(lengths[b].item())
            if valid_len == 0:
                continue
            packed = x[b, :valid_len].unsqueeze(0)  # (1, N_valid, D)
            gru_out, _ = self.gru(packed)  # (1, N_valid, 2*H)
            outputs[b, :valid_len] = self.proj(gru_out[0])
        return outputs


class STGNNPredictor(nn.Module):
    """
    时空图神经网络余震预测模型。

    架构:
    1. Event feature projection → node embeddings
    2. K 层 SpatialGraphConv + TemporalGRU blocks
    3. Global mean pooling over nodes
    4. Fusion with handcrafted global features
    5. MLP head → [target_max_mag, target_time_to_max_days]

    对应 project_plan 第 4.2 节 ST-GNN 方案。
    """

    def __init__(
        self,
        event_feature_dim: int,
        global_feature_dim: int,
        node_hidden_dim: int = 64,
        num_gnn_layers: int = 3,
        gnn_sigma: float = 50.0,
        gru_hidden_dim: int = 64,
        gru_layers: int = 2,
        global_hidden_dim: int = 128,
        fusion_hidden_dim: int = 128,
        output_dim: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.node_proj = nn.Sequential(
            nn.Linear(event_feature_dim, node_hidden_dim),
            nn.LayerNorm(node_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.gnn_blocks = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.gnn_blocks.append(
                nn.ModuleDict({
                    "spatial": SpatialGraphConv(node_hidden_dim, node_hidden_dim, sigma=gnn_sigma),
                    "temporal": TemporalGRU(node_hidden_dim, gru_hidden_dim, num_layers=gru_layers),
                    "norm1": nn.LayerNorm(node_hidden_dim),
                    "norm2": nn.LayerNorm(node_hidden_dim),
                    "dropout": nn.Dropout(dropout),
                })
            )

        self.global_encoder = nn.Sequential(
            nn.Linear(global_feature_dim, global_hidden_dim),
            nn.LayerNorm(global_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        fusion_in = node_hidden_dim + global_hidden_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, output_dim),
            nn.Softplus(),
        )

    def forward(
        self,
        seq_x: torch.Tensor,
        global_x: torch.Tensor,
        seq_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            seq_x: (B, N, event_feature_dim)
            global_x: (B, global_feature_dim)
            seq_padding_mask: (B, N) True = padding
        Returns:
            (B, 2) predictions
        """
        B, N, _ = seq_x.shape

        # Extract spatial coordinates from seq_x
        # seq_x columns: [dt_days, log_dt_days, rel_x_km, rel_y_km, distance_km, depth, mag]
        coords = seq_x[:, :, 2:4]  # (B, N, 2) — rel_x_km, rel_y_km

        # Node projection
        h = self.node_proj(seq_x)  # (B, N, node_hidden_dim)

        # GNN blocks
        for block in self.gnn_blocks:
            h_res = h
            h = block["spatial"](h, coords, seq_padding_mask)
            h = block["norm1"](h + h_res)
            h = block["dropout"](h)

            h_res = h
            h_temp = block["temporal"](h, seq_padding_mask)
            h = block["norm2"](h_res + h_temp)
            h = block["dropout"](h)

        # Global mean pool over nodes (mask-aware)
        valid_mask = (~seq_padding_mask).float().unsqueeze(-1)  # (B, N, 1)
        summed = (h * valid_mask).sum(dim=1)  # (B, node_hidden_dim)
        denom = valid_mask.sum(dim=1).clamp_min(1.0)
        seq_repr = summed / denom  # (B, node_hidden_dim)

        # Global features
        global_repr = self.global_encoder(global_x)  # (B, global_hidden_dim)

        # Fusion
        fused = torch.cat([seq_repr, global_repr], dim=-1)
        return self.fusion_head(fused)


def stgnn_asymmetric_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    late_weight: float = 2.0,
) -> torch.Tensor:
    """非对称 MSE 损失：震级 MSE + 非对称时间 MSE。"""
    mag_loss = (preds[:, 0] - targets[:, 0]).pow(2).mean()
    time_err = preds[:, 1] - targets[:, 1]
    time_weight = torch.where(
        time_err > 0,
        torch.full_like(time_err, late_weight),
        torch.ones_like(time_err),
    )
    time_loss = (time_weight * time_err.pow(2)).mean()
    return mag_loss + time_loss
