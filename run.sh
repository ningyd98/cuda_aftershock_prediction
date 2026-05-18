#!/usr/bin/env bash
# ============================================================
#  余震预测技术国际大赛 —— 一键运行脚本
#
#  推荐用法:
#    ./run.sh --skip-download              # 稳定版：LightGBM + XGBoost
#    ./run.sh --skip-download --with-dl    # 额外训练 Transformer
#    ./run.sh --skip-download --with-gnn   # 额外训练 ST-GNN
#    ./run.sh --skip-download --with-deep  # 同时训练 Transformer + ST-GNN
#    ./run.sh --no-install                 # 跳过 pip install（依赖已就绪）
#    ./run.sh --install                    # 强制重新安装依赖
#    ./run.sh train-only                   # 只重训模型并重新生成提交
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_PREFIX="${SCRIPT_DIR}/.conda"
if [ -x "${CONDA_PREFIX}/bin/python" ]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
else
    PYTHON="$(command -v python3 || command -v python)"
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
step()  { echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; echo -e "${BLUE}[STEP]${NC} $*"; echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }

show_usage() {
    cat <<'EOF'
用法:
  ./run.sh [选项] [train-only]

选项:
  --skip-download   跳过数据下载
  --skip-dl         兼容旧参数，等价于 --skip-download
  --with-dl         训练 Transformer，并在产物存在时参与融合
  --with-gnn        训练 ST-GNN，并在产物存在时参与融合
  --with-deep       同时开启 --with-dl 和 --with-gnn
  --with-gcmt       下载/使用 Global CMT 震源机制解
  --mock-eval       运行模拟线上评测
  train-only        只重训模型并重新生成提交，不重建序列和特征
EOF
}

SKIP_DOWNLOAD=false
TRAIN_ONLY=false
WITH_DL=false
WITH_GNN=false
WITH_GCMT=false
MOCK_EVAL=false
NO_INSTALL=false
FORCE_INSTALL=false

for arg in "$@"; do
    case "$arg" in
        --skip-download) SKIP_DOWNLOAD=true ;;
        --skip-dl)
            warn "--skip-dl 已弃用，请改用 --skip-download；本次仍按跳过下载处理。"
            SKIP_DOWNLOAD=true
            ;;
        --with-dl) WITH_DL=true ;;
        --with-gnn) WITH_GNN=true ;;
        --with-deep) WITH_DL=true; WITH_GNN=true ;;
        --with-gcmt) WITH_GCMT=true ;;
        --mock-eval) MOCK_EVAL=true ;;
        --no-install) NO_INSTALL=true ;;
        --install) FORCE_INSTALL=true ;;
        train-only) TRAIN_ONLY=true ;;
        -h|--help) show_usage; exit 0 ;;
        *) error "未知参数: $arg" ;;
    esac
done

M6_CATALOG="data/raw/USGS_Mw6.0_Depth70_1970-2023.csv"
FULL_CATALOG_M40="data/raw/USGS_Mw4.0_Depth70_1970-2023.csv"
FULL_CATALOG_M45="data/raw/USGS_Mw4.5_Depth70_1970-2023.csv"
ADVANCED_FEATURES="data/processed/advanced_features.csv"
MODEL_DIR="data/models"
TEST_DIR="data/test_sequences"
SUBMISSION_CSV="data/processed/submission.csv"

pick_full_catalog() {
    if [ -f "$FULL_CATALOG_M40" ]; then
        echo "$FULL_CATALOG_M40"
    elif [ -f "$FULL_CATALOG_M45" ]; then
        echo "$FULL_CATALOG_M45"
    else
        echo "$M6_CATALOG"
    fi
}

step "0. 环境检查"
"${PYTHON}" --version
info "Python: ${PYTHON}"

step "1. 安装 Python 依赖"
if [ "$FORCE_INSTALL" = true ]; then
    info "强制重新安装依赖..."
    "${PYTHON}" -m pip install -r requirements.txt
elif [ "$NO_INSTALL" = true ]; then
    info "跳过依赖安装 (--no-install)"
else
    info "检查依赖..."
    "${PYTHON}" -c "import numpy, pandas, scipy, yaml, joblib, tqdm, requests, sklearn, lightgbm, xgboost, torch" 2>/dev/null && \
        info "依赖已就绪 ✓" || \
        { warn "依赖缺失，自动安装..."; "${PYTHON}" -m pip install -r requirements.txt -q; info "依赖安装完成 ✓"; }
fi

if [ "$TRAIN_ONLY" = false ] && [ "$SKIP_DOWNLOAD" = false ]; then
    step "2. 下载数据"

    if [ ! -f data/raw/PB2002_boundaries.json ]; then
        info "下载 PB2002 板块边界..."
        "${PYTHON}" scripts/download_pb2002.py
    else
        info "PB2002 板块边界已存在，跳过 ✓"
    fi

    if [ ! -f "$M6_CATALOG" ]; then
        info "下载 USGS Mw≥6.0 主震目录..."
        "${PYTHON}" scripts/download_usgs.py
    else
        info "Mw≥6.0 主震目录已存在，跳过 ✓"
    fi

    if [ ! -f "$FULL_CATALOG_M40" ]; then
        warn "未找到 Mw≥4.0 完整事件目录，开始下载。"
        "${PYTHON}" scripts/download_full_catalog.py --min-mag 4.0
    else
        info "Mw≥4.0 完整事件目录已存在，跳过 ✓"
    fi

    if [ "$WITH_GCMT" = true ]; then
        GCMT_CATALOG="data/raw/GlobalCMT_1976-2024.csv"
        if [ ! -f "$GCMT_CATALOG" ]; then
            info "下载 Global CMT 震源机制解目录..."
            "${PYTHON}" scripts/download_gcmt.py --start-year 1976 --end-year 2024
        else
            info "Global CMT 目录已存在，跳过 ✓"
        fi
    fi
else
    info "跳过数据下载步骤"
fi

if [ "$TRAIN_ONLY" = false ]; then
    step "3. 构建主震-余震序列"
    CATALOG_FOR_SEQ="$(pick_full_catalog)"
    [ -f "$CATALOG_FOR_SEQ" ] || error "缺少事件目录: $CATALOG_FOR_SEQ"
    info "使用事件目录: ${CATALOG_FOR_SEQ}"
    "${PYTHON}" src/data_loader.py \
        --input "$CATALOG_FOR_SEQ" \
        --output data/processed/ML_Ready_Sequences.csv \
        --obs-days 3.0 \
        --target-days 30.0 \
        --radius-km 100.0
    info "序列构建完成 ✓"

    step "4. 提取高级地震学特征"
    "${PYTHON}" scripts/build_features.py --config configs/default.yaml
    info "特征提取完成 ✓"
else
    [ -f "$ADVANCED_FEATURES" ] || error "train-only 需要已存在高级特征: ${ADVANCED_FEATURES}"
    info "train-only 模式：跳过序列构建与特征生成"
fi

step "5. 训练 LightGBM + XGBoost 树模型"
"${PYTHON}" scripts/train_baseline.py \
    --data "$ADVANCED_FEATURES" \
    --n-splits 5 \
    --n-estimators 300 \
    --learning-rate 0.03 \
    --model-type both \
    --use-asymmetric-time-objective \
    --save-dir "$MODEL_DIR"
info "树模型训练完成 ✓"

DL_CATALOG="$(pick_full_catalog)"
if [ "$WITH_DL" = true ]; then
    step "5b. 训练 Transformer 深度模型"
    "${PYTHON}" scripts/train_dl.py \
        --features "$ADVANCED_FEATURES" \
        --event-catalog "$DL_CATALOG" \
        --epochs 50 \
        --batch-size 32 \
        --save-dir "$MODEL_DIR" \
        --device cpu
    info "Transformer 训练完成 ✓"
fi

if [ "$WITH_GNN" = true ]; then
    step "5c. 训练 ST-GNN 深度模型"
    "${PYTHON}" scripts/train_gnn.py \
        --features "$ADVANCED_FEATURES" \
        --event-catalog "$DL_CATALOG" \
        --epochs 50 \
        --batch-size 16 \
        --save-dir "$MODEL_DIR" \
        --device cpu
    info "ST-GNN 训练完成 ✓"
fi

step "5d. 更新可用模型融合权重"
WITH_DL_ENV="$WITH_DL" WITH_GNN_ENV="$WITH_GNN" "${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

model_dir = Path("data/models")
weights_path = model_dir / "ensemble_weights.json"
weights = {"baseline": 1.0, "xgboost": 0.0, "dl": 0.0, "gnn": 0.0}
if weights_path.exists():
    weights.update(json.loads(weights_path.read_text(encoding="utf-8")))

with_dl = os.environ["WITH_DL_ENV"] == "true"
with_gnn = os.environ["WITH_GNN_ENV"] == "true"
dl_ready = with_dl and (model_dir / "dl_model.pt").exists() and (model_dir / "dl_meta.json").exists()
gnn_ready = with_gnn and (model_dir / "gnn_model.pt").exists() and (model_dir / "gnn_meta.json").exists()

weights["dl"] = 0.0
weights["gnn"] = 0.0
deep_models = int(dl_ready) + int(gnn_ready)
if deep_models:
    deep_total = 0.15 if deep_models == 1 else 0.20
    tree_total = 1.0 - deep_total
    tree_sum = max(float(weights.get("baseline", 0.0)) + float(weights.get("xgboost", 0.0)), 1e-12)
    weights["baseline"] = round(float(weights.get("baseline", 0.0)) / tree_sum * tree_total, 4)
    weights["xgboost"] = round(float(weights.get("xgboost", 0.0)) / tree_sum * tree_total, 4)
    if dl_ready:
        weights["dl"] = round(deep_total / deep_models, 4)
    if gnn_ready:
        weights["gnn"] = round(deep_total / deep_models, 4)

weights_path.write_text(json.dumps(weights, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(weights, ensure_ascii=False, indent=2))
PY

step "6. 对测试序列生成余震预测"
shopt -s nullglob
test_files=("${TEST_DIR}"/*_eq.csv)
[ "${#test_files[@]}" -gt 0 ] || error "未找到测试序列: ${TEST_DIR}/*_eq.csv"

SUBMISSION_DIR="data/processed/submissions"
mkdir -p "$SUBMISSION_DIR"
ALL_PREDS="${SUBMISSION_DIR}/all_predictions.csv"
rm -f "$ALL_PREDS"

FIRST=true
for test_csv in "${test_files[@]}"; do
    seq_name="$(basename "$test_csv" _eq.csv)"
    out_csv="${SUBMISSION_DIR}/${seq_name}_pred.csv"
    "${PYTHON}" scripts/make_submission.py \
        --input "$test_csv" \
        --output "$out_csv" \
        --model-dir "$MODEL_DIR" \
        --allow-rule-fallback

    if [ "$FIRST" = true ]; then
        cat "$out_csv" >> "$ALL_PREDS"
        FIRST=false
    else
        tail -n +2 "$out_csv" >> "$ALL_PREDS"
    fi
done

cp "$ALL_PREDS" "$SUBMISSION_CSV"
EXPECTED_ROWS="${#test_files[@]}" SUBMISSION_CSV="$SUBMISSION_CSV" "${PYTHON}" - <<'PY'
import os
import pandas as pd

path = os.environ["SUBMISSION_CSV"]
expected_rows = int(os.environ["EXPECTED_ROWS"])
df = pd.read_csv(path)
required = ["mainshock_id", "predicted_max_mag", "predicted_time_to_max"]
missing = [col for col in required if col not in df.columns]
assert not missing, f"submission 缺少列: {missing}"
assert len(df) == expected_rows, f"submission 行数错误: {len(df)} != {expected_rows}"
assert df["mainshock_id"].is_unique, "mainshock_id 存在重复"
assert df["predicted_max_mag"].notna().all(), "预测震级存在 NaN"
assert df["predicted_time_to_max"].notna().all(), "预测时间存在 NaN"
assert (df["predicted_time_to_max"] >= 0).all(), "预测时间存在负值"
print(f"submission 校验通过: {path}, rows={len(df)}")
PY
info "汇总预测已保存: ${SUBMISSION_CSV}"

if [ "$MOCK_EVAL" = true ]; then
    step "7. 模拟线上评测"
    "${PYTHON}" scripts/mock_evaluation.py \
        --data "$ADVANCED_FEATURES" \
        --model-dir "$MODEL_DIR" \
        --output data/processed/mock_eval_report.csv \
        --stride 200 \
        --min-train-samples 500
    info "模拟线上评测完成 ✓"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║              全流程执行完毕                             ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  序列样本:   data/processed/ML_Ready_Sequences.csv      ║"
echo "║  高级特征:   data/processed/advanced_features.csv       ║"
echo "║  模型目录:   data/models                                ║"
echo "║  交叉验证:   data/models/cv_metrics.csv                 ║"
echo "║  预测提交:   data/processed/submission.csv              ║"
echo "╚══════════════════════════════════════════════════════════╝"
