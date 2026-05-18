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
    fit_dataset_preprocessors,
)
from src.models_gnn import STGNNPredictor, stgnn_asymmetric_loss
from src.utils import set_random_seed

TARGET_COLS = ["target_max_mag", "target_time_to_max_days"]
TIME_COL = "mainshock_time"
FEATURE_PREFIXES = (
    "early_", "gr_", "omori_", "anisotropy_", "plate_type_",
    "count_", "energy_", "etas_", "bath_", "fault_type_", "productivity_",
)
EXPLICIT_FEATURES = {
    "mainshock_mag", "mainshock_depth", "advanced_early_event_count",
    "plate_boundary_distance_km",
    "strike1", "dip1", "rake1", "strike2", "dip2", "rake2",
    "plunge_P", "trend_P", "plunge_T", "trend_T", "f_clvd",
    "gcmt_time_diff_seconds", "gcmt_distance_km",
    "focal_mechanism_valid",
}
EXCLUDE_COLS = {
    "mainshock_id", "mainshock_time", "mainshock_lat", "mainshock_lon",
    "nearest_plate_boundary_type", *TARGET_COLS,
}


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def select_global_feature_cols(df: pd.DataFrame) -> list[str]:
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


def train_one_epoch(model, dataloader, optimizer, device, late_weight=2.0):
    model.train()
    total_loss, n_batches = 0.0, 0
    for batch in dataloader:
        seq_x = batch["seq_x"].to(device)
        global_x = batch["global_x"].to(device)
        graph_coords_km = batch["graph_coords_km"].to(device)
        graph_time_days = batch["graph_time_days"].to(device)
        y = batch["y"].to(device)
        mask = batch["seq_padding_mask"].to(device)
        optimizer.zero_grad()
        preds = model(
            seq_x,
            global_x,
            mask,
            graph_coords_km=graph_coords_km,
            graph_time_days=graph_time_days,
        )
        loss = stgnn_asymmetric_loss(preds, y, late_weight=late_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_model(model, dataloader, device, late_weight=2.0):
    model.eval()
    all_preds, all_targets = [], []
    for batch in dataloader:
        seq_x = batch["seq_x"].to(device)
        global_x = batch["global_x"].to(device)
        graph_coords_km = batch["graph_coords_km"].to(device)
        graph_time_days = batch["graph_time_days"].to(device)
        y = batch["y"].to(device)
        mask = batch["seq_padding_mask"].to(device)
        preds = model(
            seq_x,
            global_x,
            mask,
            graph_coords_km=graph_coords_km,
            graph_time_days=graph_time_days,
        )
        all_preds.append(preds.cpu().numpy())
        all_targets.append(y.cpu().numpy())
    preds_arr = np.clip(np.concatenate(all_preds, axis=0), a_min=0.0, a_max=None)
    targets_arr = np.concatenate(all_targets, axis=0)
    preds_eval = preds_arr.copy()
    targets_eval = targets_arr.copy()
    preds_eval[:, 1] = np.expm1(np.clip(preds_eval[:, 1], 0.0, 50.0))
    targets_eval[:, 1] = np.expm1(np.clip(targets_eval[:, 1], 0.0, 50.0))
    from src.evaluator import calculate_metrics
    return calculate_metrics(
        y_true_mag=targets_eval[:, 0], y_pred_mag=preds_eval[:, 0],
        y_true_time=targets_eval[:, 1], y_pred_time=preds_eval[:, 1],
        late_weight=late_weight,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="训练 ST-GNN 余震预测模型")
    parser.add_argument("--features", type=Path, default=PROJECT_ROOT / "data" / "processed" / "advanced_features.csv")
    parser.add_argument("--event-catalog", type=Path, default=PROJECT_ROOT / "data" / "raw" / "USGS_Mw4.5_Depth70_1970-2023.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--node-hidden", type=int, default=64)
    parser.add_argument("--gnn-layers", type=int, default=3)
    parser.add_argument("--gnn-radius-km", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=Path, default=PROJECT_ROOT / "data" / "models")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    features_path = resolve_project_path(args.features)
    event_catalog_path = resolve_project_path(args.event_catalog)
    if not event_catalog_path.exists():
        event_catalog_path = PROJECT_ROOT / "data" / "raw" / "USGS_Mw6.0_Depth70_1970-2023.csv"

    df = pd.read_csv(features_path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True, errors="coerce")
    df = df.dropna(subset=[TIME_COL, *TARGET_COLS]).sort_values(TIME_COL).reset_index(drop=True)

    event_df = pd.read_csv(event_catalog_path)
    event_df["time"] = pd.to_datetime(event_df["time"], utc=True, errors="coerce")

    global_cols = select_global_feature_cols(df)
    seq_config = SequenceBuildConfig(obs_days=3.0, spatial_radius_km=100.0, max_seq_len=256)

    n_val = max(1, int(len(df) * 0.2))
    train_df = df.iloc[:-n_val].copy()
    val_df = df.iloc[-n_val:].copy()
    preprocessors = fit_dataset_preprocessors(
        sequence_df=train_df,
        event_catalog_df=event_df,
        global_feature_cols=global_cols,
        target_cols=TARGET_COLS,
        config=seq_config,
        scaler_type="robust",
        add_missing_indicators=True,
    )

    train_set = EarthquakeSequenceDataset(
        sequence_df=train_df, event_catalog_df=event_df,
        global_feature_cols=global_cols, target_cols=TARGET_COLS, config=seq_config,
        preprocessors=preprocessors, fit_preprocessors=False,
    )
    val_set = EarthquakeSequenceDataset(
        sequence_df=val_df, event_catalog_df=event_df,
        global_feature_cols=global_cols, target_cols=TARGET_COLS, config=seq_config,
        preprocessors=preprocessors, fit_preprocessors=False,
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=earthquake_collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=earthquake_collate_fn)

    print(
        f"训练: {len(train_set)}, 验证: {len(val_set)}, "
        f"全局特征: {len(global_cols)}, 全局输入维: {train_set.global_feature_dim}, "
        f"缺失指示列: {len(train_set.preprocessors.global_indicator_cols)}, "
        f"事件特征: {len(train_set.event_feature_cols)}"
    )

    model = STGNNPredictor(
        event_feature_dim=len(train_set.event_feature_cols),
        global_feature_dim=train_set.global_feature_dim,
        node_hidden_dim=args.node_hidden,
        num_gnn_layers=args.gnn_layers,
        gnn_radius_km=args.gnn_radius_km,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_mag_rmse = float("inf")
    save_dir = resolve_project_path(args.save_dir)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate_model(model, val_loader, device)
        scheduler.step()

        if val_metrics["mag_rmse"] < best_mag_rmse:
            best_mag_rmse = val_metrics["mag_rmse"]
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / "gnn_model.pt")
            preprocessor_path = save_dir / "gnn_preprocessors.joblib"
            joblib.dump(train_set.preprocessors, preprocessor_path)
            meta = {
                "model_class": "STGNNPredictor",
                "event_feature_dim": len(train_set.event_feature_cols),
                "global_feature_dim": train_set.global_feature_dim,
                "global_feature_cols": global_cols,
                "global_indicator_cols": train_set.preprocessors.global_indicator_cols,
                "preprocessor_path": preprocessor_path.name,
                "time_target_transform": "log1p",
                "node_hidden_dim": args.node_hidden,
                "num_gnn_layers": args.gnn_layers,
                "gnn_radius_km": args.gnn_radius_km,
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "best_val_metrics": val_metrics,
            }
            with (save_dir / "gnn_meta.json").open("w") as f:
                json.dump(meta, f, indent=2)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {train_loss:.4f} | MagRMSE: {val_metrics['mag_rmse']:.3f} | TimeRMSE: {val_metrics['time_rmse']:.3f}")

    print(f"\n最佳 MagRMSE: {best_mag_rmse:.4f}, 模型: {save_dir / 'gnn_model.pt'}")


if __name__ == "__main__":
    main()
