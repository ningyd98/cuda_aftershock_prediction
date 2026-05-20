#!/usr/bin/env bash
# ============================================================
#  一键调优 + 训练 + 提交脚本
#  用法: ./run_tuning.sh
#        ./run_tuning.sh --tune-only   # 只调优，不训练
#        ./run_tuning.sh --train-only  # 用已有最优参数训练
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

CONDA_PREFIX="${SCRIPT_DIR}/.conda"
if [ -x "${CONDA_PREFIX}/bin/python" ]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
else
    PYTHON="$(command -v python3 || command -v python)"
fi

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
step()  { echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; echo -e "${BLUE}[STEP]${NC} $*"; echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }

ADVANCED_FEATURES="data/processed/advanced_features.csv"
MODEL_DIR="data/models"
TEST_DIR="data/test_sequences"

TUNE_ONLY=false
TRAIN_ONLY=false
N_TRIALS=100

for arg in "$@"; do
    case "$arg" in
        --tune-only) TUNE_ONLY=true ;;
        --train-only) TRAIN_ONLY=true ;;
        *) error "未知参数: $arg" ;;
    esac
done

# ====== 前置检查 ======
[ -f "$ADVANCED_FEATURES" ] || error "缺少高级特征: $ADVANCED_FEATURES"
[ -d "$TEST_DIR" ] && [ "$(ls "$TEST_DIR"/*_eq.csv 2>/dev/null | wc -l)" -gt 0 ] || \
    warn "未找到测试序列，将跳过提交生成"

# ====== Phase 1: 超参数调优 ======
if [ "$TRAIN_ONLY" = false ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    TUNE_BASE="data/tuning_results/tune_${TIMESTAMP}"

    # ------ Step 1a: 树模型粗调 (100 trials, 无 DL/GNN, 约 30-60 分钟) ------
    step "Phase 1a: 树模型超参数调优 (LightGBM + XGBoost, ${N_TRIALS} trials)"
    info "输出: ${TUNE_BASE}_trees/"
    "${PYTHON}" scripts/tune_all_models.py \
        --n-trials "$N_TRIALS" \
        --n-estimators 300 \
        --late-weight 2.0 \
        --mag-weight 3.0 \
        --study-name "aftershock_trees_${TIMESTAMP}" \
        --output-dir "${TUNE_BASE}_trees" \
        --eval-holdout-every 10 \
        --device cuda \
        --no-dl --no-gnn
    TREE_BEST_PARAMS="${TUNE_BASE}_trees/best_params.json"
    TREE_WEIGHTS="${TUNE_BASE}_trees/ensemble_weights.json"
    info "树模型调优完成！最优参数: ${TREE_BEST_PARAMS}"

    # ------ Step 1b: 全模型快速调优 (50 trials, 含 DL+GNN, 约 2-4 小时) ------
    step "Phase 1b: 全模型快速调优 (LGB+XGB+DL+GNN, 50 trials, 减少 DL/GNN epochs)"
    info "输出: ${TUNE_BASE}_full/"
    "${PYTHON}" scripts/tune_all_models.py \
        --n-trials 50 \
        --n-estimators 300 \
        --late-weight 2.0 \
        --mag-weight 3.0 \
        --study-name "aftershock_full_${TIMESTAMP}" \
        --output-dir "${TUNE_BASE}_full" \
        --eval-holdout-every 10 \
        --device cuda \
        --fast
    FULL_BEST_PARAMS="${TUNE_BASE}_full/best_params.json"
    FULL_WEIGHTS="${TUNE_BASE}_full/ensemble_weights.json"
    info "全模型调优完成！最优参数: ${FULL_BEST_PARAMS}"

    # ------ 用最优权重覆盖模型目录 ------
    mkdir -p "$MODEL_DIR"
    if [ -f "$FULL_WEIGHTS" ]; then
        cp "$FULL_WEIGHTS" "$MODEL_DIR/ensemble_weights.json"
        info "已将全模型最优融合权重复制到 ${MODEL_DIR}/ensemble_weights.json"
    elif [ -f "$TREE_WEIGHTS" ]; then
        cp "$TREE_WEIGHTS" "$MODEL_DIR/ensemble_weights.json"
        info "已将树模型最优融合权重复制到 ${MODEL_DIR}/ensemble_weights.json"
    fi

    if [ "$TUNE_ONLY" = true ]; then
        info "调优完成！产物:"
        info "  树模型: ${TUNE_BASE}_trees/"
        info "  全模型: ${TUNE_BASE}_full/"
        exit 0
    fi
fi

# ====== Phase 2: OOF 全流程训练 ======
step "Phase 2: OOF 全流程训练 (树∥DL → GNN → 融合 → 全量训练)"

DL_CATALOG="data/raw/USGS_Mw4.5_Depth70_1970-2023.csv"
[ -f "$DL_CATALOG" ] || DL_CATALOG="data/raw/USGS_Mw4.0_Depth70_1970-2023.csv"
[ -f "$DL_CATALOG" ] || DL_CATALOG="data/raw/USGS_Mw6.0_Depth70_1970-2023.csv"

# ------ Step 2a: 树 + DL OOF 并行 ------
step "2a. 树模型 + Transformer OOF CV (并行)"
"${PYTHON}" scripts/train_baseline.py \
    --data "$ADVANCED_FEATURES" \
    --n-splits 5 \
    --purge-days 30.0 \
    --n-estimators 500 \
    --learning-rate 0.02 \
    --model-type both \
    --use-asymmetric-time-objective \
    --device cuda \
    --save-dir "$MODEL_DIR" &
TREE_PID=$!

"${PYTHON}" scripts/train_dl.py \
    --features "$ADVANCED_FEATURES" \
    --event-catalog "$DL_CATALOG" \
    --epochs 30 \
    --batch-size 64 \
    --num-workers 4 \
    --save-dir "$MODEL_DIR" \
    --device cuda \
    --oof --n-splits 5 --purge-days 30.0 \
    --oof-output "$MODEL_DIR/dl_oof_predictions.csv" &
DL_PID=$!

wait $TREE_PID && info "树模型 OOF 完成"
wait $DL_PID   && info "DL OOF 完成"

# ------ Step 2b: GNN OOF ------
step "2b. ST-GNN OOF CV"
"${PYTHON}" scripts/train_gnn.py \
    --features "$ADVANCED_FEATURES" \
    --event-catalog "$DL_CATALOG" \
    --epochs 30 \
    --batch-size 32 \
    --num-workers 4 \
    --no-torch-compile \
    --save-dir "$MODEL_DIR" \
    --device cuda \
    --oof --n-splits 5 --purge-days 30.0 \
    --oof-output "$MODEL_DIR/gnn_oof_predictions.csv"
info "GNN OOF 完成"

# ------ Step 2c: 融合权重搜索 ------
step "2c. 多模型融合权重搜索"
"${PYTHON}" scripts/train_ensemble.py \
    --model-dir "$MODEL_DIR" \
    --grid-step 0.02
info "融合权重已更新: ${MODEL_DIR}/ensemble_weights.json"

# ------ Step 2d: Full-Fit 全量训练 (DL ∥ GNN) ------
step "2d. 全量训练最终 Transformer + ST-GNN (并行)"
"${PYTHON}" scripts/train_dl.py \
    --features "$ADVANCED_FEATURES" \
    --event-catalog "$DL_CATALOG" \
    --epochs 50 \
    --batch-size 64 \
    --num-workers 4 \
    --save-dir "$MODEL_DIR" \
    --device cuda &
DL_FULL_PID=$!

"${PYTHON}" scripts/train_gnn.py \
    --features "$ADVANCED_FEATURES" \
    --event-catalog "$DL_CATALOG" \
    --epochs 50 \
    --batch-size 32 \
    --num-workers 4 \
    --no-torch-compile \
    --save-dir "$MODEL_DIR" \
    --device cuda &
GNN_FULL_PID=$!

wait $DL_FULL_PID  && info "最终 Transformer 已保存"
wait $GNN_FULL_PID && info "最终 ST-GNN 已保存"

# ====== Phase 3: 生成提交 ======
step "Phase 3: 对测试序列生成预测提交"

shopt -s nullglob
test_files=("${TEST_DIR}"/*_eq.csv)
if [ "${#test_files[@]}" -gt 0 ]; then
    SUBMISSION_DIR="data/processed/submissions"
    mkdir -p "$SUBMISSION_DIR"
    ALL_PREDS="${SUBMISSION_DIR}/all_predictions.csv"
    rm -f "$ALL_PREDS"
    FIRST=true
    GATING_ARGS=()
    [ -f "${MODEL_DIR}/aftershock_classifier.joblib" ] && GATING_ARGS=(--use-gating)

    for test_csv in "${test_files[@]}"; do
        seq_name="$(basename "$test_csv" _eq.csv)"
        out_csv="${SUBMISSION_DIR}/${seq_name}_pred.csv"
        "${PYTHON}" scripts/make_submission.py \
            --input "$test_csv" \
            --output "$out_csv" \
            --model-dir "$MODEL_DIR" \
            "${GATING_ARGS[@]}" \
            --allow-rule-fallback 2>/dev/null || true
        if [ "$FIRST" = true ]; then
            cat "$out_csv" >> "$ALL_PREDS"; FIRST=false
        else
            tail -n +2 "$out_csv" >> "$ALL_PREDS" 2>/dev/null || true
        fi
    done
    cp "$ALL_PREDS" "data/processed/submission.csv"
    info "提交已保存: data/processed/submission.csv"
fi

step "全流程调优+训练完成！"
echo ""
echo "产物清单:"
echo "  树模型 OOF:     ${MODEL_DIR}/oof_predictions.csv"
echo "  DL OOF:         ${MODEL_DIR}/dl_oof_predictions.csv"
echo "  GNN OOF:        ${MODEL_DIR}/gnn_oof_predictions.csv"
echo "  融合权重:       ${MODEL_DIR}/ensemble_weights.json"
echo "  CV 指标:        ${MODEL_DIR}/cv_metrics.csv"
echo "  最终模型:       ${MODEL_DIR}/baseline_model.joblib, xgboost_model.joblib"
echo "                  ${MODEL_DIR}/dl_model.pt, ${MODEL_DIR}/gnn_model.pt"
echo "  提交文件:       data/processed/submission.csv"
echo "  调优结果:       ${TUNE_BASE:-data/tuning_results/}_trees/"
echo "                  ${TUNE_BASE:-data/tuning_results/}_full/"
