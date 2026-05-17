#!/usr/bin/env bash
# ============================================================
#  余震预测技术国际大赛 —— 一键运行脚本
#  run.sh
#
#  用法:
#    chmod +x run.sh
#    ./run.sh              # 全流程：下载 → 建序列 → 特征 → 训练 → 预测
#    ./run.sh --skip-dl    # 跳过下载（假设数据已就绪）
#    ./run.sh --with-dl    # 同时训练 Transformer + ST-GNN 深度模型
#    ./run.sh --with-gcmt  # 额外下载 Global CMT 震源机制解数据
#    ./run.sh --mock-eval  # 额外运行模拟线上评测
#    ./run.sh train-only   # 仅重新训练
# ============================================================

set -euo pipefail

# --------------------- 项目根目录 ---------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Conda 环境前缀
CONDA_PREFIX="${SCRIPT_DIR}/.conda"
PYTHON="${CONDA_PREFIX}/bin/python"
CONDA_RUN="conda run -p ${CONDA_PREFIX} --no-capture-output"

# --------------------- 颜色 ---------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
step()  { echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; echo -e "${BLUE}[STEP]${NC} $*"; echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }

# --------------------- 参数解析 ---------------------
SKIP_DOWNLOAD=false
TRAIN_ONLY=false
WITH_DL=false
WITH_GCMT=false
MOCK_EVAL=false

for arg in "$@"; do
    case "$arg" in
        --skip-dl) SKIP_DOWNLOAD=true ;;
        --with-dl) WITH_DL=true ;;
        --with-gcmt) WITH_GCMT=true ;;
        --mock-eval) MOCK_EVAL=true ;;
        train-only) TRAIN_ONLY=true ;; 
        train-only) TRAIN_ONLY=true ;;
        *) error "未知参数: $arg" ;;
    esac
done

# --------------------- Step 0: 环境检查 ---------------------
step "0. 环境检查"

if [ ! -d "$CONDA_PREFIX" ]; then
    error "Conda 环境未找到: ${CONDA_PREFIX}\n请先创建环境: conda create -p ${CONDA_PREFIX} python=3.11 -y"
fi

# 检查 llvm-openmp (LightGBM 在 macOS 上的依赖)
if [[ "$(uname)" == "Darwin" ]]; then
    if ! ${PYTHON} -c "import ctypes; ctypes.cdll.LoadLibrary('@rpath/libomp.dylib')" &>/dev/null; then
        warn "安装 LightGBM macOS 依赖: llvm-openmp ..."
        conda install -p "${CONDA_PREFIX}" -c conda-forge llvm-openmp -y
    fi
fi

info "环境检查通过 ✓"

# --------------------- Step 1: 安装依赖 ---------------------
step "1. 安装 Python 依赖"

${PYTHON} -m pip install -r requirements.txt -q 2>&1 | tail -1
info "依赖安装完成 ✓"

# --------------------- Step 2: 下载数据 ---------------------
if [ "$TRAIN_ONLY" = false ] && [ "$SKIP_DOWNLOAD" = false ]; then
    step "2. 下载数据"

    # 2a. 板块边界 (PB2002)
    if [ ! -f data/raw/PB2002_boundaries.json ]; then
        info "下载板块边界数据..."
        ${PYTHON} scripts/download_pb2002.py
    else
        info "PB2002 板块边界已存在，跳过 ✓"
    fi

    # 2b. USGS Mw≥6.0 强震目录 (主震识别用)
    M6_CATALOG="data/raw/USGS_Mw6.0_Depth70_1970-2023.csv"
    if [ ! -f "$M6_CATALOG" ]; then
        info "下载 USGS Mw≥6.0 强震目录..."
        ${PYTHON} scripts/download_usgs.py
    else
        info "Mw≥6.0 强震目录已存在，跳过 ✓"
    fi

    # 2c. USGS Mw≥4.5 完整目录 (余震特征提取用)
    FULL_CATALOG="data/raw/USGS_Mw4.5_Depth70_1970-2023.csv"
    if [ ! -f "$FULL_CATALOG" ]; then
        warn "未找到完整事件目录 (Mw≥4.5)。"
        warn "该目录用于计算 G-R b 值、大森定律等高级特征。"
        warn "正在下载... (预计 1-2 分钟，取决于网络)"
        ${PYTHON} scripts/download_full_catalog.py
    else
        info "完整事件目录已存在，跳过 ✓"
    fi

    # 2d. Global CMT 震源机制解目录 (可选)
    if [ "$WITH_GCMT" = true ]; then
        GCMT_CATALOG="data/raw/GlobalCMT_1976-2024.csv"
        if [ ! -f "$GCMT_CATALOG" ]; then
            warn "正在下载 Global CMT 震源机制解目录..."
            warn "这可能需要几分钟（下载 576+ 个月文件）"
            ${PYTHON} scripts/download_gcmt.py --start-year 1976 --end-year 2024
        else
            info "GCMT 目录已存在，跳过 ✓"
        fi
    fi
else
    info "跳过数据下载步骤"
fi

# --------------------- Step 3: 构建主震-余震序列 ---------------------
step "3. 构建主震-余震序列"

FULL_CATALOG="data/raw/USGS_Mw4.5_Depth70_1970-2023.csv"
M6_CATALOG="data/raw/USGS_Mw6.0_Depth70_1970-2023.csv"

# 优先使用完整目录构建序列（含低震级余震），回退到 Mw≥6.0 目录
if [ -f "$FULL_CATALOG" ]; then
    CATALOG_FOR_SEQ="$FULL_CATALOG"
    info "使用完整目录构建序列 (Mw≥4.5)"
else
    CATALOG_FOR_SEQ="$M6_CATALOG"
    warn "仅 Mw≥6.0 目录可用，余震特征将大量缺失"
fi

${PYTHON} src/data_loader.py \
    --input "$CATALOG_FOR_SEQ" \
    --output data/processed/ML_Ready_Sequences.csv \
    --obs-days 3.0 --target-days 30.0 --radius-km 100.0

info "序列构建完成 ✓"

# --------------------- Step 4: 提取高级特征 ---------------------
step "4. 提取高级地震学特征 (并行)"

# 特征提取使用完整目录或回退目录
if [ -f "$FULL_CATALOG" ]; then
    FEATURE_CATALOG="$FULL_CATALOG"
else
    FEATURE_CATALOG="$M6_CATALOG"
fi

${PYTHON} scripts/build_features.py \
    --config configs/default.yaml

info "特征提取完成 ✓"

# --------------------- Step 5: 训练 Baseline 模型 ---------------------
step "5. 训练 LightGBM + XGBoost 双模型"

${PYTHON} scripts/train_baseline.py \
    --data data/processed/advanced_features.csv \
    --n-splits 5 \
    --n-estimators 300 \
    --learning-rate 0.03 \
    --model-type both \
    --save-dir data/models

info "双模型训练完成 ✓"

# --------------------- Step 5b: 训练深度学习模型 (可选) ---------------------
if [ "$WITH_DL" = true ]; then
    step "5b. 训练深度学习模型 (Transformer + ST-GNN)"

    FULL_CATALOG="data/raw/USGS_Mw4.5_Depth70_1970-2023.csv"
    M6_CATALOG="data/raw/USGS_Mw6.0_Depth70_1970-2023.csv"
    if [ -f "$FULL_CATALOG" ]; then
        DL_CATALOG="$FULL_CATALOG"
    else
        DL_CATALOG="$M6_CATALOG"
    fi

    # Transformer
    info "训练 Transformer 模型..."
    ${PYTHON} scripts/train_dl.py \
        --features data/processed/advanced_features.csv \
        --event-catalog "$DL_CATALOG" \
        --epochs 50 \
        --batch-size 32 \
        --save-dir data/models \
        --device cpu

    # ST-GNN
    info "训练 ST-GNN 模型..."
    ${PYTHON} scripts/train_gnn.py \
        --features data/processed/advanced_features.csv \
        --event-catalog "$DL_CATALOG" \
        --epochs 50 \
        --batch-size 16 \
        --save-dir data/models \
        --device cpu

    # 更新融合权重
    ${PYTHON} -c "
import json
weights = {'baseline': 0.5, 'xgboost': 0.3, 'dl': 0.2}
with open('data/models/ensemble_weights.json', 'w') as f:
    json.dump(weights, f, indent=2)
print('融合权重已更新: baseline=0.5, xgboost=0.3, dl=0.2')
"
    info "深度学习模型训练完成 ✓"
fi

# --------------------- Step 6: 对测试序列生成预测 ---------------------
step "6. 对测试序列生成余震预测"

# 为每条测试序列单独生成预测，汇总为一个 submission
SUBMISSION_DIR="data/processed/submissions"
mkdir -p "$SUBMISSION_DIR"

TEST_DIR="data/test_sequences"
ALL_PREDS="${SUBMISSION_DIR}/all_predictions.csv"
> "$ALL_PREDS"

FIRST=true
for test_csv in "$TEST_DIR"/*_eq.csv; do
    SEQ_NAME=$(basename "$test_csv" _eq.csv)
    OUT_CSV="${SUBMISSION_DIR}/${SEQ_NAME}_pred.csv"
    
    ${PYTHON} scripts/make_submission.py \
        --input "$test_csv" \
        --output "$OUT_CSV" \
        --baseline-model data/models/baseline_model.joblib \
        --feature-cols data/models/feature_cols.json \
        --ensemble-weights data/models/ensemble_weights.json \
        --allow-rule-fallback 2>/dev/null || true
    
    if [ -f "$OUT_CSV" ]; then
        if [ "$FIRST" = true ]; then
            cat "$OUT_CSV" >> "$ALL_PREDS"
            FIRST=false
        else
            tail -n +2 "$OUT_CSV" >> "$ALL_PREDS" 2>/dev/null || true
        fi
    fi
done

# 最终 submission
SUBMISSION_CSV="data/processed/submission.csv"
if [ -f "$ALL_PREDS" ] && [ "$(wc -l < "$ALL_PREDS")" -gt 1 ]; then
    cp "$ALL_PREDS" "$SUBMISSION_CSV"
    info "汇总预测已保存: ${SUBMISSION_CSV}"
else
    warn "未生成有效预测，请检查模型产物是否已训练完成"
fi

# --------------------- Step 7: 模拟线上评测 (可选) ---------------------
if [ "$MOCK_EVAL" = true ]; then
    step "7. 模拟线上评测"

    ${PYTHON} scripts/mock_evaluation.py \
        --data data/processed/advanced_features.csv \
        --model-dir data/models \
        --output data/processed/mock_eval_report.csv \
        --stride 200 \
        --min-train-samples 500

    info "模拟线上评测完成 ✓"
    info "评测报告: data/processed/mock_eval_report.csv"
fi

# --------------------- 完成 ---------------------
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║         🎉  全流程执行完毕！                            ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  序列样本:   data/processed/ML_Ready_Sequences.csv      ║"
echo "║  高级特征:   data/processed/advanced_features.csv       ║"
echo "║  训练模型:   data/models/baseline_model.joblib          ║"
echo "║  交叉验证:   data/models/cv_metrics.csv                 ║"
echo "║  预测提交:   data/processed/submission.csv              ║"
echo "╚══════════════════════════════════════════════════════════╝"
