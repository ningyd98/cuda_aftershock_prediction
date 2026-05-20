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
    - 权重 = exp(-d²/(2σ²)) · exp(magnitude_alpha · (M_source - Mc))
      震级越大的 source 事件对邻域影响越大，对应 ETAS 模型的物理机制
    - 按 target 行归一化
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        sigma: float = 50.0,
        radius_km: float = 100.0,
        mc: float = 2.5,
    ) -> None:
        super().__init__()
        self.sigma = sigma
        self.radius_km = radius_km
        self.mc = mc
        self.linear = nn.Linear(in_dim, out_dim)
        # 可学习的震级影响系数（softplus 保证非负）
        self.magnitude_alpha_raw = nn.Parameter(torch.tensor(0.5))

    @property
    def magnitude_alpha(self) -> torch.Tensor:
        """Softplus 约束保证 alpha >= 0。"""
        return torch.nn.functional.softplus(self.magnitude_alpha_raw)

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        event_times: torch.Tensor | None,
        mask: torch.Tensor,
        event_mags: torch.Tensor | None = None,
        strike_rad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D_in) node features
            coords: (B, N, 2) node spatial coordinates in km
            event_times: (B, N) elapsed days after mainshock
            mask: (B, N) True = padding
            event_mags: (B, N) event magnitudes for magnitude-aware edge weights (optional)
            strike_rad: (B,) mainshock strike angle in radians (optional, for anisotropic edges)
        Returns:
            (B, N, D_out) updated node features
        """
        B, N, _ = x.shape

        coord_diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # (B, N, N, 2)
        dists = torch.norm(coord_diff, dim=-1)  # (B, N, N)

        adj = torch.exp(-(dists ** 2) / (2 * self.sigma ** 2))  # (B, N, N)
        adj = adj * (dists < self.radius_km).float()

        # ─── 各向异性应力传导：沿断层走向强化，垂直走向抑制 ───
        if strike_rad is not None:
            strike_rad = strike_rad.to(x.device)
            # 计算边方向 azimuth: atan2(dx, dy) where dx=east, dy=north
            dx = coord_diff[..., 0]  # (B, N, N)
            dy = coord_diff[..., 1]
            edge_azimuth = torch.atan2(dx, dy)  # (B, N, N), range [-π, π]
            # 与走向的夹角
            theta = edge_azimuth - strike_rad.view(B, 1, 1)  # (B, N, N)
            # cos(2θ): 沿走向最大=1, 垂直最小=-1 → (1+cos)/2 ∈ [0,1]
            aniso_weight = (1.0 + torch.cos(2.0 * theta)) * 0.5
            adj = adj * aniso_weight

        # ─── 震级感知边权重：对应 ETAS 触发能力 ∝ exp(α·(M - Mc)) ───
        if event_mags is not None:
            # source 震级 (B, 1, N) 对每条边的 source 端加权
            source_mags = event_mags.unsqueeze(1)  # (B, 1, N)
            mag_weight = torch.exp(
                self.magnitude_alpha * (source_mags - self.mc)
            )  # (B, 1, N)
            adj = adj * mag_weight  # broadcast: (B, N, N) * (B, 1, N) → (B, N, N)

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
        self.proj = nn.Linear(hidden_dim * 2, in_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) event features
            mask: (B, N) True = padding
        Returns:
            (B, N, D_hidden) temporal encoding per node
        """
        lengths = (~mask).sum(dim=1).cpu().clamp_min(1)  # (B,) 防止长度为 0

        # 完全向量化的 GRU 处理，消除 for 循环瓶颈
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        gru_out, _ = self.gru(packed)
        unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(
            gru_out, batch_first=True, total_length=x.shape[1]
        )
        
        # 投影并应用 mask 清理填充区域的输出
        outputs = self.proj(unpacked)
        valid_mask = (~mask).unsqueeze(-1).float()
        return outputs * valid_mask


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
            nn.ReLU(),
        )

    def forward(
        self,
        seq_x: torch.Tensor,
        global_x: torch.Tensor,
        seq_padding_mask: torch.Tensor,
        graph_coords_km: torch.Tensor | None = None,
        graph_time_days: torch.Tensor | None = None,
        graph_strike_rad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            seq_x: (B, N, event_feature_dim)
            global_x: (B, global_feature_dim)
            seq_padding_mask: (B, N) True = padding
            graph_coords_km: (B, N, 2) 未标准化空间坐标，单位 km
            graph_time_days: (B, N) 未标准化相对时间，单位天
            graph_strike_rad: (B,) 主震走向弧度（可选，用于各向异性边）
        Returns:
            (B, 2) predictions
        """
        B, N, _ = seq_x.shape

        # 兼容旧调用：优先使用 Dataset 提供的真实物理坐标/时间。
        coords = graph_coords_km if graph_coords_km is not None else seq_x[:, :, 2:4]
        event_times = graph_time_days if graph_time_days is not None else seq_x[:, :, 0]
        # 提取事件震级用于震级感知边权重（seq_x 最后一维为 mag）
        event_mags = seq_x[:, :, -1]  # (B, N)，未标准化的原始震级

        # Node projection
        h = self.node_proj(seq_x)  # (B, N, node_hidden_dim)

        # GNN blocks
        for block in self.gnn_blocks:
            h_res = h
            h = block["spatial"](h, coords, event_times, seq_padding_mask,
                                  event_mags=event_mags, strike_rad=graph_strike_rad)
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
