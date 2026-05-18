from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.dataset import (
    EarthquakeSequenceDataset,
    SequenceBuildConfig,
    earthquake_collate_fn,
)
from src.models_dl import Seq2SeqAftershockPredictor, asymmetric_time_mse_loss
from src.utils import set_random_seed


TARGET_COLS = ["target_max_mag", "target_time_to_max_days"]
TIME_COL = "mainshock_time"
FEATURE_PREFIXES = (
    "early_",
    "gr_",
    "omori_",
    "anisotropy_",
    "plate_type_",
    "count_",
    "energy_",
    "etas_",
    "bath_",
    "fault_type_",
    "productivity_",
)
EXPLICIT_FEATURES = {
    "mainshock_mag",
    "mainshock_depth",
    "advanced_early_event_count",
    "plate_boundary_distance_km",
    "strike1",
    "dip1",
    "rake1",
    "strike2",
    "dip2",
    "rake2",
    "plunge_P",
    "trend_P",
    "plunge_T",
    "trend_T",
    "f_clvd",
    "focal_mechanism_valid",
}
EXCLUDE_COLS = {
    "mainshock_id",
    "mainshock_time",
    "mainshock_lat",
    "mainshock_lon",
    "nearest_plate_boundary_type",
    *TARGET_COLS,
}


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def select_global_feature_cols(df: pd.DataFrame) -> list[str]:
    """筛选数值型全局特征（与 LightGBM 训练保持一致）。"""
    candidates: list[str] = []
    for col in df.columns:
        if col in EXCLUDE_COLS:
            continue
        if col in EXPLICIT_FEATURES or col.startswith(FEATURE_PREFIXES):
            if pd.api.types.is_bool_dtype(df[col]):
                df[col] = df[col].astype(int)
            if pd.api.types.is_numeric_dtype(df[col]):
                candidates.append(col)
    return candidates


def train_one_epoch(
    model: Seq2SeqAftershockPredictor,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    late_weight: float = 2.0,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        seq_x = batch["seq_x"].to(device)
        global_x = batch["global_x"].to(device)
        y = batch["y"].to(device)
        seq_padding_mask = batch["seq_padding_mask"].to(device)

        optimizer.zero_grad()
        preds = model(seq_x, global_x, seq_padding_mask)
        loss = asymmetric_time_mse_loss(preds, y, late_weight=late_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_model(
    model: Seq2SeqAftershockPredictor,
    dataloader: DataLoader,
    device: torch.device,
    late_weight: float = 2.0,
) -> dict:
    model.eval()
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for batch in dataloader:
        seq_x = batch["seq_x"].to(device)
        global_x = batch["global_x"].to(device)
        y = batch["y"].to(device)
        seq_padding_mask = batch["seq_padding_mask"].to(device)

        preds = model(seq_x, global_x, seq_padding_mask)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(y.cpu().numpy())

    preds_arr = np.concatenate(all_preds, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)
    preds_arr = np.clip(preds_arr, a_min=0.0, a_max=None)

    from src.evaluator import calculate_metrics

    return calculate_metrics(
        y_true_mag=targets_arr[:, 0],
        y_pred_mag=preds_arr[:, 0],
        y_true_time=targets_arr[:, 1],
        y_pred_time=preds_arr[:, 1],
        late_weight=late_weight,
    )


def time_series_train_val_split(
    df: pd.DataFrame,
    val_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按时间排序后取最后 val_ratio 作为验证集。"""
    n = len(df)
    n_val = max(1, int(n * val_ratio))
    return df.iloc[:-n_val].copy(), df.iloc[-n_val:].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练余震预测深度学习模型 (Transformer)")
    parser.add_argument(
        "--features",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "advanced_features.csv",
    )
    parser.add_argument(
        "--event-catalog",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "USGS_Mw4.5_Depth70_1970-2023.csv",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=Path, default=PROJECT_ROOT / "data" / "models")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载数据
    features_path = resolve_project_path(args.features)
    event_catalog_path = resolve_project_path(args.event_catalog)

    if not event_catalog_path.exists():
        fallback = PROJECT_ROOT / "data" / "raw" / "USGS_Mw6.0_Depth70_1970-2023.csv"
        print(f"⚠ 完整目录不存在，回退到: {fallback}")
        event_catalog_path = fallback

    df = pd.read_csv(features_path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True, errors="coerce")
    df = df.dropna(subset=[TIME_COL, *TARGET_COLS]).sort_values(TIME_COL).reset_index(drop=True)
    event_df = pd.read_csv(event_catalog_path)
    event_df["time"] = pd.to_datetime(event_df["time"], utc=True, errors="coerce")

    global_cols = select_global_feature_cols(df)

    seq_config = SequenceBuildConfig(obs_days=3.0, spatial_radius_km=100.0, max_seq_len=256)

    train_df, val_df = time_series_train_val_split(df, val_ratio=0.2)
    train_dataset = EarthquakeSequenceDataset(
        sequence_df=train_df,
        event_catalog_df=event_df,
        global_feature_cols=global_cols,
        target_cols=TARGET_COLS,
        config=seq_config,
        fit_preprocessors=True,
        scaler_type="robust",
    )
    val_dataset = EarthquakeSequenceDataset(
        sequence_df=val_df,
        event_catalog_df=event_df,
        global_feature_cols=global_cols,
        target_cols=TARGET_COLS,
        config=seq_config,
        preprocessors=train_dataset.preprocessors,
        fit_preprocessors=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=earthquake_collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=earthquake_collate_fn, num_workers=0,
    )

    print(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")
    print(f"全局特征数: {len(global_cols)}")
    print(f"全局输入维: {train_dataset.global_feature_dim}")
    print(f"缺失指示列数: {len(train_dataset.preprocessors.global_indicator_cols)}")
    print(f"事件特征维: {len(train_dataset.event_feature_cols)}")

    model = Seq2SeqAftershockPredictor(
        event_feature_dim=len(train_dataset.event_feature_cols),
        global_feature_dim=train_dataset.global_feature_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_mag_rmse = float("inf")
    save_dir = resolve_project_path(args.save_dir)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate_model(model, val_loader, device)
        scheduler.step()

        if val_metrics["mag_rmse"] < best_val_mag_rmse:
            best_val_mag_rmse = val_metrics["mag_rmse"]
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / "dl_model.pt")
            preprocessor_path = save_dir / "dl_preprocessors.joblib"
            joblib.dump(train_dataset.preprocessors, preprocessor_path)
            dl_meta = {
                "model_class": "Seq2SeqAftershockPredictor",
                "event_feature_dim": len(train_dataset.event_feature_cols),
                "global_feature_dim": train_dataset.global_feature_dim,
                "global_feature_cols": global_cols,
                "global_indicator_cols": train_dataset.preprocessors.global_indicator_cols,
                "preprocessor_path": preprocessor_path.name,
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_layers": args.num_layers,
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "best_val_metrics": val_metrics,
            }
            with (save_dir / "dl_meta.json").open("w", encoding="utf-8") as f:
                json.dump(dl_meta, f, ensure_ascii=False, indent=2)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"Val MagRMSE: {val_metrics['mag_rmse']:.3f} | "
                f"Val TimeRMSE: {val_metrics['time_rmse']:.3f}"
            )

    print(f"\n最佳验证 Mag RMSE: {best_val_mag_rmse:.4f}")
    print(f"模型已保存: {save_dir / 'dl_model.pt'}")


if __name__ == "__main__":
    main()
