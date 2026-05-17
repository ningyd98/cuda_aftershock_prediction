from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """标准 Transformer 正弦位置编码。"""

    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        positions = torch.arange(max_len).unsqueeze(1)
        div_terms = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(positions * div_terms)
        pe[:, 1::2] = torch.cos(positions * div_terms[: pe[:, 1::2].shape[1]])

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError(
                f"序列长度 {x.size(1)} 超过位置编码最大长度 {self.pe.size(1)}。"
            )
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MaskedMeanPool(nn.Module):
    """对 Transformer 输出做 mask-aware mean pooling。"""

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        valid_mask = (~padding_mask).unsqueeze(-1).float()
        summed = (x * valid_mask).sum(dim=1)
        denom = valid_mask.sum(dim=1).clamp_min(1.0)
        return summed / denom


class Seq2SeqAftershockPredictor(nn.Module):
    """
    双输入融合模型。

    早期余震事件序列经 TransformerEncoder 编码，全局手工特征经 MLP 编码，
    两路表示拼接后预测 [target_max_mag, target_time_to_max_days]。
    """

    def __init__(
        self,
        event_feature_dim: int,
        global_feature_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        global_hidden_dim: int = 128,
        fusion_hidden_dim: int = 128,
        output_dim: int = 2,
        max_seq_len: int = 256,
    ) -> None:
        super().__init__()

        self.event_projection = nn.Sequential(
            nn.Linear(event_feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.position_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=max_seq_len,
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.sequence_pool = MaskedMeanPool()

        self.global_encoder = nn.Sequential(
            nn.Linear(global_feature_dim, global_hidden_dim),
            nn.LayerNorm(global_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(d_model + global_hidden_dim, fusion_hidden_dim),
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
        参数:
            seq_x: (batch, seq_len, event_feature_dim)
            global_x: (batch, global_feature_dim)
            seq_padding_mask: (batch, seq_len)，True 表示 padding
        """
        seq_embed = self.event_projection(seq_x)
        seq_embed = self.position_encoding(seq_embed)

        # PyTorch attention 不喜欢某个样本所有 token 都被 mask。
        # 空序列样本临时放开第一个零 token，pooling 仍用原始 mask 得到零向量。
        encoder_padding_mask = seq_padding_mask.clone()
        empty_sequence_mask = encoder_padding_mask.all(dim=1)
        if empty_sequence_mask.any():
            encoder_padding_mask[empty_sequence_mask, 0] = False

        encoded_seq = self.transformer(
            seq_embed,
            src_key_padding_mask=encoder_padding_mask,
        )
        seq_repr = self.sequence_pool(encoded_seq, seq_padding_mask)
        global_repr = self.global_encoder(global_x)

        fused = torch.cat([seq_repr, global_repr], dim=-1)
        return self.fusion_head(fused)


def asymmetric_time_mse_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    late_weight: float = 2.0,
    mag_weight: float = 1.0,
    time_weight: float = 1.0,
) -> torch.Tensor:
    """训练用损失：震级 MSE + 非对称时间 MSE。"""
    pred_mag = preds[:, 0]
    pred_time = preds[:, 1]
    true_mag = targets[:, 0]
    true_time = targets[:, 1]

    mag_loss = (pred_mag - true_mag).pow(2).mean()

    time_error = pred_time - true_time
    time_weights = torch.where(
        time_error > 0,
        torch.full_like(time_error, late_weight),
        torch.ones_like(time_error),
    )
    time_loss = (time_weights * time_error.pow(2)).mean()

    return mag_weight * mag_loss + time_weight * time_loss
