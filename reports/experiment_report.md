# Qualification Experiment Report

Generated from the remote CUDA run on 2026-05-21 (Asia/Shanghai). The run targets the qualification submission format: T1 (0-24h), T2 (24-72h), and T3 (72-168h). Each test sequence produces `{YYYYMMDDhhmmss}-T1-T2.csv` and `{YYYYMMDDhhmmss}-T3.csv` without headers, using the official space-separated sample format.

## Remote Environment

- Host: `ningyd-Ubuntu`
- GPU: NVIDIA GeForce RTX 4070 Ti SUPER
- Driver/CUDA reported by `nvidia-smi`: 595.58.03 / CUDA 13.2
- Python: 3.12.7
- Libraries: NumPy 2.4.4, pandas 3.0.3, scikit-learn 1.8.0, LightGBM 4.6.0, XGBoost 3.2.0, PyTorch 2.12.0+cu130

## Data And Targets

- Source feature table: `data/processed/advanced_features.csv`
- Qualification feature table: `data/processed/qualification_features.csv`
- Training rows: 4711
- Target columns: `target_T1_max_mag`, `target_T1_time_to_max_hours`, `target_T2_max_mag`, `target_T2_time_to_max_hours`, `target_T3_max_mag`, `target_T3_time_to_max_hours`
- Feature count used by the window models: 109

## Reproduction Commands

```bash
python main.py build-qualification-labels \
  --catalog data/raw/USGS_Mw4.0_Depth70_1970-2023.csv \
  --base-features data/processed/advanced_features.csv \
  --output data/processed/qualification_features.csv

python main.py train-window-baseline \
  --data data/processed/qualification_features.csv \
  --n-splits 5 \
  --seed 42 \
  --device cuda \
  --model-type both \
  --n-estimators 500 \
  --learning-rate 0.03 \
  --save-dir data/models/qualification_best
```

## Tuning Summary

The tuning proxy score is the mean over available window/model metrics of `mag_rmse + time_hour_asymmetric_rmse / 24`. Lower is better.

| Rank | Config | Proxy Score | Mean Mag RMSE | Mean Asym Time RMSE (h) | Mean Hit Rate |
|--:|:--|--:|--:|--:|--:|
| 1 | both_500_lr003 | 1.3548 | 0.8448 | 12.2384 | 78.59% |
| 2 | both_800_lr002 | 1.3548 | 0.8443 | 12.2523 | 78.64% |
| 3 | both_500_lr005 | 1.3669 | 0.8538 | 12.3136 | 78.58% |
| 4 | lgbm_800_lr002 | 1.3697 | 0.8487 | 12.5026 | 78.61% |
| 5 | lgbm_500_lr003 | 1.3708 | 0.8513 | 12.4687 | 78.58% |
| 6 | both_500_lr003_asym2 | 1.3747 | 0.8449 | 12.7161 | 78.33% |
| 7 | lgbm_500_lr005 | 1.3824 | 0.8580 | 12.5847 | 78.56% |
| 8 | lgbm_500_lr003_asym2 | 1.4109 | 0.8513 | 13.4294 | 78.11% |

Best configuration: `both_500_lr003`.

## Best Window Metrics

| Window | Model | Mag RMSE | Mag MAE | Time RMSE (h) | Time MAE (h) | Asym Time RMSE (h) | Hit Rate |
|:--|:--|--:|--:|--:|--:|--:|--:|
| T1 | baseline | 0.1887 | 0.0611 | 3.0665 | 1.5990 | 3.3571 | 84.20% |
| T1 | xgboost | 0.1866 | 0.0587 | 3.0409 | 1.5854 | 3.3474 | 84.54% |
| T2 | baseline | 0.3398 | 0.1024 | 9.5294 | 5.5482 | 11.1064 | 73.38% |
| T2 | xgboost | 0.3396 | 0.1013 | 9.2521 | 5.2116 | 10.9361 | 72.84% |
| T3 | baseline | 2.0253 | 1.5267 | 19.2155 | 12.9059 | 22.9216 | 78.22% |
| T3 | xgboost | 1.9889 | 1.5074 | 18.4135 | 11.9226 | 21.7616 | 78.37% |

## Prediction Package Check

Using the best model, a no-commitment inspection package was generated on the remote host at `qualification_submission_no_commitment.zip`.

- Test sequences: 20
- Prediction files: 40
- Format validation: passed, with T1/T2 files containing 2 rows and T3 files containing 1 row
- SHA256: `eba3a9a60290d73ae1318b64d40c606ab99c662fdc4ed9675a0baecf779ac403`

The official final ZIP should still include the official commitment letter if the competition platform enforces the original qualification package checklist.
