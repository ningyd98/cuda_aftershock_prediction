from __future__ import annotations

import math

import torch
from torch import nn


class SpatialGraphConv(nn.Module):
    """
    有向时空图卷积层：距离近且时间更早的事件向未来事件传递信息。

    邻接矩阵使用 A[target, source] 约定：
    - 距离 d(target, source) < radius_km
    - time[source] < time[target]
    - 权重为 exp(-d² / (2σ²))，按 target 行归一化
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        sigma: float = 50.0,
        radius_km: float = 100.0,
    ) -> None:
        super().__init__()
        self.sigma = sigma
        self.radius_km = radius_km
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        event_times: torch.Tensor | None,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D_in) node features
            coords: (B, N, 2) node spatial coordinates in km
            event_times: (B, N) elapsed days after mainshock
            mask: (B, N) True = padding
        Returns:
            (B, N, D_out) updated node features
        """
        B, N, _ = x.shape

        coord_diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # (B, N, N, 2)
        dists = torch.norm(coord_diff, dim=-1)  # (B, N, N)

        adj = torch.exp(-(dists ** 2) / (2 * self.sigma ** 2))  # (B, N, N)
        adj = adj * (dists < self.radius_km).float()

        if event_times is not None:
            # A[target, source]，只允许过去 source 指向未来 target。
            source_times = event_times.unsqueeze(1)
            target_times = event_times.unsqueeze(2)
            causal_mask = source_times < target_times
            adj = adj * causal_mask.float()
        else:
            # 兼容旧调用：没有真实时间时至少去掉自环。
            eye = torch.eye(N, device=x.device, dtype=torch.float32).unsqueeze(0)
            adj = adj * (1.0 - eye)

        valid_nodes = (~mask).float()
        valid_target = valid_nodes.unsqueeze(2)
        valid_source = valid_nodes.unsqueeze(1)
        adj = adj * valid_target * valid_source

        deg = adj.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        adj_norm = adj / deg
        messages = self.linear(x)  # (B, N, D_out)
        return torch.bmm(adj_norm, messages)  # target rows aggregate source columns


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
        gnn_radius_km: float = 100.0,
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
                    "spatial": SpatialGraphConv(
                        node_hidden_dim,
                        node_hidden_dim,
                        sigma=gnn_sigma,
                        radius_km=gnn_radius_km,
                    ),
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
        graph_coords_km: torch.Tensor | None = None,
        graph_time_days: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            seq_x: (B, N, event_feature_dim)
            global_x: (B, global_feature_dim)
            seq_padding_mask: (B, N) True = padding
            graph_coords_km: (B, N, 2) 未标准化空间坐标，单位 km
            graph_time_days: (B, N) 未标准化相对时间，单位天
        Returns:
            (B, 2) predictions
        """
        B, N, _ = seq_x.shape

        # 兼容旧调用：优先使用 Dataset 提供的真实物理坐标/时间。
        coords = graph_coords_km if graph_coords_km is not None else seq_x[:, :, 2:4]
        event_times = graph_time_days if graph_time_days is not None else seq_x[:, :, 0]

        # Node projection
        h = self.node_proj(seq_x)  # (B, N, node_hidden_dim)

        # GNN blocks
        for block in self.gnn_blocks:
            h_res = h
            h = block["spatial"](h, coords, event_times, seq_padding_mask)
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
