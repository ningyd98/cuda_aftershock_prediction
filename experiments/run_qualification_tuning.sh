#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/ningyd/CodingSpace/aftershock_qualification_train
PY=/home/ningyd/CodingSpace/aftershock_prediction/.conda/bin/python
RUN_ID=${RUN_ID:-qualification_tuning_$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ROOT="$PROJECT/experiments/$RUN_ID"
LOG="$RUN_ROOT/train.log"
mkdir -p "$RUN_ROOT/models" "$RUN_ROOT/reports"

exec > >(tee -a "$LOG") 2>&1

echo "RUN_ID=$RUN_ID"
echo "START_UTC=$(date -u --iso-8601=seconds)"
echo "PROJECT=$PROJECT"
cd "$PROJECT"

echo "SYSTEM"
hostname
nvidia-smi

echo "VERSIONS"
"$PY" - <<'PY'
import platform
mods = ['numpy','pandas','sklearn','joblib','lightgbm','xgboost','torch']
print('python', platform.python_version())
for m in mods:
    mod = __import__(m)
    print(m, getattr(mod, '__version__', 'ok'))
try:
    import torch
    print('torch_cuda_available', torch.cuda.is_available())
    print('torch_cuda_device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
except Exception as exc:
    print('torch_cuda_probe_error', exc)
PY

echo "COMPILE"
"$PY" -m compileall main.py src scripts tests

echo "BUILD_LABELS"
"$PY" main.py build-qualification-labels \
  --catalog data/raw/USGS_Mw4.0_Depth70_1970-2023.csv \
  --base-features data/processed/advanced_features.csv \
  --output data/processed/qualification_features.csv

run_config() {
  local name="$1"
  shift
  local out="$RUN_ROOT/models/$name"
  mkdir -p "$out"
  echo "CONFIG_START $name $(date -u --iso-8601=seconds)"
  /usr/bin/time -f "ELAPSED_SECONDS %e" "$PY" main.py train-window-baseline \
    --data data/processed/qualification_features.csv \
    --n-splits 5 \
    --seed 42 \
    --device cuda \
    --save-dir "$out" \
    "$@"
  "$PY" scripts/generate_experiment_report.py \
    --metrics "$out/qualification_window_metrics.json" \
    --output "$RUN_ROOT/reports/${name}.md" || true
  echo "CONFIG_END $name $(date -u --iso-8601=seconds)"
}

run_config lgbm_500_lr003 --model-type lightgbm --n-estimators 500 --learning-rate 0.03
run_config lgbm_800_lr002 --model-type lightgbm --n-estimators 800 --learning-rate 0.02
run_config lgbm_500_lr005 --model-type lightgbm --n-estimators 500 --learning-rate 0.05
run_config lgbm_500_lr003_asym2 --model-type lightgbm --n-estimators 500 --learning-rate 0.03 --use-asymmetric-time-objective --late-weight 2.0
run_config both_500_lr003 --model-type both --n-estimators 500 --learning-rate 0.03
run_config both_800_lr002 --model-type both --n-estimators 800 --learning-rate 0.02
run_config both_500_lr005 --model-type both --n-estimators 500 --learning-rate 0.05
run_config both_500_lr003_asym2 --model-type both --n-estimators 500 --learning-rate 0.03 --use-asymmetric-time-objective --late-weight 2.0

echo "SUMMARIZE"
"$PY" - <<'PY'
from pathlib import Path
import json, shutil
import pandas as pd

project = Path('/home/ningyd/CodingSpace/aftershock_qualification_train')
run_root = sorted((project / 'experiments').glob('qualification_tuning_*'))[-1]
rows = []
for metrics_path in sorted((run_root / 'models').glob('*/qualification_window_metrics.json')):
    config = metrics_path.parent.name
    payload = json.loads(metrics_path.read_text(encoding='utf-8'))
    for window, models in payload.get('window_metrics', {}).items():
        for model, m in models.items():
            score_proxy = float(m['mag_rmse']) + float(m['time_hour_asymmetric_rmse']) / 24.0
            rows.append({
                'config': config,
                'window': window,
                'model': model,
                'mag_rmse': m.get('mag_rmse'),
                'mag_mae': m.get('mag_mae'),
                'time_hour_rmse': m.get('time_hour_rmse'),
                'time_hour_mae': m.get('time_hour_mae'),
                'time_hour_asymmetric_rmse': m.get('time_hour_asymmetric_rmse'),
                'time_hour_hit_rate': m.get('time_hour_hit_rate'),
                'score_proxy': score_proxy,
            })
summary = pd.DataFrame(rows)
summary.to_csv(run_root / 'qualification_tuning_summary_by_window.csv', index=False)
config_summary = (
    summary.groupby('config', as_index=False)
    .agg(
        score_proxy=('score_proxy', 'mean'),
        mag_rmse=('mag_rmse', 'mean'),
        time_hour_asymmetric_rmse=('time_hour_asymmetric_rmse', 'mean'),
        time_hour_hit_rate=('time_hour_hit_rate', 'mean'),
    )
    .sort_values('score_proxy')
)
config_summary.to_csv(run_root / 'qualification_tuning_summary.csv', index=False)
best = config_summary.iloc[0].to_dict()
(run_root / 'best_config.json').write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding='utf-8')
best_name = str(best['config'])
best_dir = run_root / 'models' / best_name
final_dir = project / 'data' / 'models' / 'qualification_best'
if final_dir.exists():
    shutil.rmtree(final_dir)
shutil.copytree(best_dir, final_dir)
print('BEST_CONFIG', json.dumps(best, ensure_ascii=False))
print('BEST_MODEL_DIR', final_dir)
PY

"$PY" scripts/generate_experiment_report.py \
  --metrics data/models/qualification_best/qualification_window_metrics.json \
  --output reports/experiment_report.md || true

echo "END_UTC=$(date -u --iso-8601=seconds)"
echo "DONE $RUN_ROOT"
