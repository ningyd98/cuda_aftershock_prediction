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
from sklearn.model_selection import TimeSeriesSplit
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
from src.utils import get_torch_device, set_random_seed, setup_cuda, try_torch_compile

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


# ============================================================
#  OOF 模式 — TimeSeriesSplit + purge
# ============================================================

def _prepare_features_for_gnn_oof(
    features_path: Path,
    event_catalog_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    df = pd.read_csv(features_path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True, errors="coerce")
    df = df.dropna(subset=[TIME_COL, *TARGET_COLS]).sort_values(TIME_COL).reset_index(drop=True)

    event_df = pd.read_csv(event_catalog_path)
    event_df["time"] = pd.to_datetime(event_df["time"], utc=True, errors="coerce")

    global_cols = select_global_feature_cols(df)
    return df, event_df, global_cols


def _run_oof_for_gnn(
    df: pd.DataFrame,
    event_df: pd.DataFrame,
    global_cols: list[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    device = setup_cuda(
        args.device,
        deterministic=getattr(args, "cuda_deterministic", False),
        allow_tf32=getattr(args, "cuda_allow_tf32", True),
        matmul_precision=getattr(args, "cuda_matmul_precision", "high"),
    )
    train_df = df.sort_values(TIME_COL).reset_index(drop=True)
    splitter = TimeSeriesSplit(n_splits=args.n_splits)
    purge_delta = pd.Timedelta(days=float(getattr(args, "purge_days", 30.0)))

    seq_config = SequenceBuildConfig(obs_days=3.0, spatial_radius_km=100.0, max_seq_len=256)
    oof_preds = np.full((len(train_df), len(TARGET_COLS)), np.nan, dtype=float)
    fold_records: list[dict] = []

    num_workers = getattr(args, "num_workers", 0)
    pin_memory = getattr(args, "pin_memory", True) and device.type == "cuda"
    use_compile = getattr(args, "use_torch_compile", False)

    fold_iter = tqdm(
        enumerate(splitter.split(train_df), start=1),
        total=args.n_splits,
        desc="GNN OOF folds",
        unit="fold",
    )
    for fold_idx, (train_idx, valid_idx) in fold_iter:
        valid_start_time = train_df.loc[valid_idx[0], TIME_COL]
        fold_iter.set_postfix(
            train=len(train_idx),
            valid=len(valid_idx),
            start=str(valid_start_time)[:10],
        )
        purge_cutoff = valid_start_time - purge_delta
        purge_mask = train_df.loc[train_idx, TIME_COL] <= purge_cutoff
        train_idx_purged = train_idx[purge_mask.values]
        if len(train_idx_purged) < max(10, len(train_idx) * 0.3):
            train_idx_purged = train_idx

        fold_train_df = train_df.iloc[train_idx_purged].copy()
        fold_valid_df = train_df.iloc[valid_idx].copy()

        preprocessors = fit_dataset_preprocessors(
            sequence_df=fold_train_df, event_catalog_df=event_df,
            global_feature_cols=global_cols, target_cols=TARGET_COLS,
            config=seq_config, scaler_type="robust", add_missing_indicators=True,
        )
        train_dataset = EarthquakeSequenceDataset(
            sequence_df=fold_train_df, event_catalog_df=event_df,
            global_feature_cols=global_cols, target_cols=TARGET_COLS,
            config=seq_config, preprocessors=preprocessors, fit_preprocessors=False,
        )
        valid_dataset = EarthquakeSequenceDataset(
            sequence_df=fold_valid_df, event_catalog_df=event_df,
            global_feature_cols=global_cols, target_cols=TARGET_COLS,
            config=seq_config, preprocessors=preprocessors, fit_preprocessors=False,
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=earthquake_collate_fn,
                                  num_workers=num_workers, pin_memory=pin_memory)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                  collate_fn=earthquake_collate_fn,
                                  num_workers=num_workers, pin_memory=pin_memory)

        model = STGNNPredictor(
            event_feature_dim=len(train_dataset.event_feature_cols),
            global_feature_dim=train_dataset.global_feature_dim,
            node_hidden_dim=args.node_hidden,
            num_gnn_layers=args.gnn_layers,
            gnn_radius_km=args.gnn_radius_km,
        ).to(device)

        if use_compile:
            model = try_torch_compile(model)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        best_val_loss = float("inf")
        best_state = None

        epoch_iter = tqdm(
            range(1, args.epochs + 1),
            desc=f"GNN Fold {fold_idx} epochs",
            unit="epoch",
            leave=False,
        )
        for _ in epoch_iter:
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                late_weight=args.late_weight,
            )
            model.eval()
            val_loss_total, val_count = 0.0, 0
            with torch.no_grad():
                for batch in valid_loader:
                    sx = batch["seq_x"].to(device)
                    gx = batch["global_x"].to(device)
                    coords = batch["graph_coords_km"].to(device)
                    gtd = batch["graph_time_days"].to(device)
                    yy = batch["y"].to(device)
                    mk = batch["seq_padding_mask"].to(device)
                    pp = model(sx, gx, mk, graph_coords_km=coords, graph_time_days=gtd)
                    val_loss_total += stgnn_asymmetric_loss(pp, yy, late_weight=args.late_weight).item() * len(yy)
                    val_count += len(yy)
            val_loss = val_loss_total / max(val_count, 1)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epoch_iter.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_loss=f"{val_loss:.4f}",
                best=f"{best_val_loss:.4f}",
            )
            scheduler.step()

        if best_state is None:
            raise RuntimeError(
                f"OOF fold {fold_idx}: 训练 {args.epochs} 轮后 best_state 仍为 None，"
                "请检查模型是否正常训练（loss 可能为 NaN/Inf）"
            )

        model.load_state_dict(best_state)
        model.eval()

        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in tqdm(
                valid_loader,
                desc=f"GNN Fold {fold_idx} OOF预测",
                unit="batch",
                leave=False,
            ):
                sx = batch["seq_x"].to(device)
                gx = batch["global_x"].to(device)
                coords = batch["graph_coords_km"].to(device)
                gtd = batch["graph_time_days"].to(device)
                yy = batch["y"].to(device)
                mk = batch["seq_padding_mask"].to(device)
                pp = model(sx, gx, mk, graph_coords_km=coords, graph_time_days=gtd)
                all_preds.append(pp.cpu().numpy())
                all_targets.append(yy.cpu().numpy())

        preds_arr = np.clip(np.concatenate(all_preds, axis=0), a_min=0.0, a_max=None)
        targets_arr = np.concatenate(all_targets, axis=0)
        preds_arr[:, 1] = np.expm1(np.clip(preds_arr[:, 1], 0.0, 50.0))
        targets_arr[:, 1] = np.expm1(np.clip(targets_arr[:, 1], 0.0, 50.0))
        oof_preds[valid_idx] = preds_arr

        from src.evaluator import calculate_metrics
        metrics = calculate_metrics(
            y_true_mag=targets_arr[:, 0], y_pred_mag=preds_arr[:, 0],
            y_true_time=targets_arr[:, 1], y_pred_time=preds_arr[:, 1],
            late_weight=args.late_weight,
        )
        fold_records.append({
            "fold": fold_idx, "model": "gnn",
            "train_size": len(train_idx_purged), "valid_size": len(valid_idx),
            "purge_days": float(getattr(args, "purge_days", 30.0)),
            "train_start": str(train_df.loc[train_idx_purged[0], TIME_COL])[:10],
            "train_end": str(train_df.loc[train_idx_purged[-1], TIME_COL])[:10],
            "valid_start": str(valid_start_time)[:10],
            "valid_end": str(train_df.loc[valid_idx[-1], TIME_COL])[:10],
            **metrics,
        })
        print(f"  GNN Fold {fold_idx}/{args.n_splits} | MagRMSE={metrics['mag_rmse']:.3f} | TimeRMSE={metrics['time_rmse']:.3f}")

    oof_df = train_df[["mainshock_id", TIME_COL, *TARGET_COLS]].copy()
    oof_df["gnn_pred_mag"] = oof_preds[:, 0]
    oof_df["gnn_pred_time"] = oof_preds[:, 1]

    fold_metrics_df = pd.DataFrame(fold_records)
    return oof_df, fold_metrics_df


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
    parser.add_argument("--device", type=str, default="auto",
                        help="设备: auto, cuda, mps, cpu")
    # OOF 模式参数
    parser.add_argument("--oof", action="store_true", help="OOF 交叉验证模式")
    parser.add_argument("--n-splits", type=int, default=5, help="OOF 折数")
    parser.add_argument("--purge-days", type=float, default=30.0, help="OOF purge 天数")
    parser.add_argument("--oof-output", type=Path, default=None, help="OOF 预测输出路径")
    parser.add_argument("--late-weight", type=float, default=2.0, help="预测偏晚惩罚权重")
    # CUDA / 性能
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--pin-memory", action="store_true", default=True, help="pin_memory 加速")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false",
                        help="禁用 pin_memory")
    parser.add_argument("--use-torch-compile", action="store_true", help="torch.compile 加速")
    parser.add_argument("--cuda-deterministic", action="store_true",
                        help="cudnn deterministic 可复现模式")
    parser.add_argument("--cuda-allow-tf32", action="store_true", default=True,
                        help="允许 TF32 (Ampere+ GPU)")
    parser.add_argument("--cuda-matmul-precision", type=str, default="high",
                        choices=["highest", "high", "medium"])
    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed)
    torch.manual_seed(args.seed)

    if args.epochs <= 0:
        raise ValueError(f"epochs 必须 > 0, 当前: {args.epochs}")

    # ---- OOF 模式 ----
    if args.oof:
        features_path = resolve_project_path(args.features)
        event_catalog_path = resolve_project_path(args.event_catalog)
        if not event_catalog_path.exists():
            event_catalog_path = PROJECT_ROOT / "data" / "raw" / "USGS_Mw6.0_Depth70_1970-2023.csv"

        oof_output = resolve_project_path(args.oof_output) if args.oof_output else (
            resolve_project_path(args.save_dir) / "gnn_oof_predictions.csv"
        )

        df, event_df, global_cols = _prepare_features_for_gnn_oof(features_path, event_catalog_path)
        print(f"GNN OOF 模式: {len(df)} 样本, {len(global_cols)} 全局特征, {args.n_splits} 折")

        oof_df, fold_metrics_df = _run_oof_for_gnn(df, event_df, global_cols, args)

        oof_output.parent.mkdir(parents=True, exist_ok=True)
        oof_df.to_csv(oof_output, index=False, encoding="utf-8")
        fold_metrics_df.to_csv(oof_output.parent / "gnn_cv_metrics.csv", index=False, encoding="utf-8")

        print(f"\nGNN OOF 预测已保存: {oof_output}")
        print(f"GNN OOF 平均指标:")
        metric_cols = [c for c in fold_metrics_df.columns if c not in {"fold","model","train_size","valid_size","purge_days","train_start","train_end","valid_start","valid_end"}]
        for col in metric_cols:
            print(f"  {col}: {fold_metrics_df[col].mean():.4f}")
        return

    # ---- 常规训练模式 ----
    device = setup_cuda(
        args.device,
        deterministic=getattr(args, "cuda_deterministic", False),
        allow_tf32=getattr(args, "cuda_allow_tf32", True),
        matmul_precision=getattr(args, "cuda_matmul_precision", "high"),
    )
    print(f"设备: {device}")

    num_workers = getattr(args, "num_workers", 0)
    pin_memory = getattr(args, "pin_memory", True) and device.type == "cuda"
    use_compile = getattr(args, "use_torch_compile", False)

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

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        collate_fn=earthquake_collate_fn,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        collate_fn=earthquake_collate_fn,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    print(
        f"训练: {len(train_set)}, 验证: {len(val_set)}, "
        f"全局特征: {len(global_cols)}, 全局输入维: {train_set.global_feature_dim}, "
        f"缺失指示列: {len(train_set.preprocessors.global_indicator_cols)}, "
        f"事件特征: {len(train_set.event_feature_cols)}"
    )
    print(f"DataLoader: num_workers={num_workers}, pin_memory={pin_memory}, torch_compile={use_compile}")

    model = STGNNPredictor(
        event_feature_dim=len(train_set.event_feature_cols),
        global_feature_dim=train_set.global_feature_dim,
        node_hidden_dim=args.node_hidden,
        num_gnn_layers=args.gnn_layers,
        gnn_radius_km=args.gnn_radius_km,
    ).to(device)

    if use_compile:
        model = try_torch_compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_mag_rmse = float("inf")
    save_dir = resolve_project_path(args.save_dir)

    epoch_iter = tqdm(range(1, args.epochs + 1), desc="GNN 训练 epochs", unit="epoch")
    for epoch in epoch_iter:
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            late_weight=args.late_weight,
        )
        val_metrics = evaluate_model(model, val_loader, device, late_weight=args.late_weight)
        scheduler.step()
        epoch_iter.set_postfix(
            loss=f"{train_loss:.4f}",
            mag_rmse=f"{val_metrics['mag_rmse']:.3f}",
            time_rmse=f"{val_metrics['time_rmse']:.3f}",
        )

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
