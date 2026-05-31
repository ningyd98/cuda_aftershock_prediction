#!/usr/bin/env bash
# Qualification submission one-click pipeline.
#
# Three pipelines available:
#   1. Hybrid calibrated   (default, formal T1/T2/T3 window output)
#   2. Decoupled           (--use-decoupled, formal-compatible per-window models)
#   3. Single-horizon H168 (--use-single-horizon, auxiliary experiment only)
#
# Default behavior builds the required T1/T2/T3 qualification package without
# a commitment letter. Pass --commitment-template for an official ZIP.
#
# Competition output format:
#   {mainshock}-T1-T2.csv  # 2 lines for T1 and T2
#   {mainshock}-T3.csv     # 1 line for T3

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

CONDA_PREFIX="${PROJECT_ROOT}/.conda"
if [ -n "${PYTHON:-}" ] && [ -x "$PYTHON" ]; then
    :  # use existing PYTHON from environment
elif [ -x "${CONDA_PREFIX}/bin/python" ]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
else
    PYTHON="$(command -v python3 || command -v python)"
fi
# Safety: ensure PYTHON is set before any command uses it
[ -z "${PYTHON:-}" ] && PYTHON="$(command -v python3)" || true
[ -n "${PYTHON:-}" ] || error "无法找到 Python 解释器。请安装 Python 3 或设置 PYTHON 环境变量。"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
step()  { echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; echo -e "${BLUE}[STEP]${NC} $*"; echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }

INPUT_DIR="data/test_sequences"
FEATURE_DATA="data/processed/qualification_features.csv"
BASE_FEATURES="data/processed/advanced_features.csv"
CATALOG=""
SCORE_MODEL=""
LEGAL_MODEL="data/models/qualification_legal_fusion/qualification_window_models.joblib"
OUTPUT_DIR="submission_package_hybrid_calibrated_no_commitment"
ZIP_PATH="qualification_submission_hybrid_calibrated_no_commitment.zip"
COMMITMENT_TEMPLATE=""
SKIP_COMMITMENT=true
CLEAN=true
NO_INSTALL=false
FORCE_INSTALL=false
BUILD_LABELS=false
TRAIN_SCORE=false
TRAIN_LEGAL=false
NO_CALIBRATION=false
DEVICE="cpu"
N_SPLITS=5
N_ESTIMATORS=500
LEARNING_RATE=0.03
MODEL_TYPE="both"

# Decoupled pipeline flags
TUNE_DECOUPLED=false
TUNE_TRIALS=80
TUNE_OBJECTIVE="balanced"
TUNE_FAST=false
MAG_TRIALS=""
TIME_TRIALS=""
EXTREME_TRIALS=""
RETRAIN_DECOUPLED=false
USE_DECOUPLED=false
DECOUPLED_MODEL_PATH="data/models/qualification_decoupled/qualification_decoupled_models.joblib"
DECOUPLED_BEST_PARAMS=""
DECOUPLED_OUTPUT_DIR="submission_package_decoupled"
DECOUPLED_ZIP_PATH="qualification_submission_decoupled.zip"

# Single-horizon H168 pipeline flags (formal route for one prediction per sequence)
TUNE_SINGLE_HORIZON=false
RETRAIN_SINGLE_HORIZON=false
USE_SINGLE_HORIZON=false
SINGLE_HORIZON_MODEL_PATH="data/models/single_horizon/qualification_single_horizon_model.joblib"
SINGLE_HORIZON_BEST_PARAMS=""
SINGLE_HORIZON_OUTPUT_DIR="submission_package_single_horizon"
SINGLE_HORIZON_ZIP_PATH="qualification_submission_single_horizon.zip"

show_usage() {
    cat <<'EOF'
用法:
  ./run.sh qualification [选项]
  scripts/run_qualification_pipeline.sh [选项]

默认动作:
  生成比赛要求的 T1/T2/T3 三窗口资格赛提交检查包：
  qualification_submission_hybrid_calibrated_no_commitment.zip

常用:
  --no-install              跳过依赖检查/安装
  --full-train              重建资格赛标签，并重训 score model 与 legal-risk model
  --retrain-score           只重训三窗口 score model
  --retrain-legal           只重训合法窗口 + 极端大余震风险模型
  --build-labels            用完整目录重建 T1/T2/T3 标签
  --no-calibration          关闭 T1/T2/T3 后校准，额外生成未校准包时使用
  --commitment-template P   加入官方承诺书模板，生成正式提交 ZIP

路径:
  --input-dir DIR           测试震例目录，默认 data/test_sequences
  --features CSV            资格赛训练特征，默认 data/processed/qualification_features.csv
  --catalog CSV             完整地震目录；缺省自动选择 Mw4.0/Mw4.5/Mw6.0 目录
  --score-model-path PATH   score model；缺省优先 data/models/qualification_best/...
  --legal-model-path PATH   legal-risk model，默认 data/models/qualification_legal_fusion/...
  --output-dir DIR          解压后的提交包目录
  --zip-path PATH           ZIP 输出路径

训练参数:
  --device cuda|cpu|auto      默认 cpu（qualification 纯 CPU 流程）
  --n-splits N
  --n-estimators N
  --learning-rate X
  --model-type lightgbm|xgboost|both

Decoupled 管道（震级/时间分离训练 + 时间桶分类）:
  --tune-decoupled          运行 decoupled 模型超参数调优
  --tune-trials N           Optuna/随机搜索 trial 数 (默认 80)
  --tune-objective OBJ      优化目标: balanced|mag|time|official_like (默认 balanced)
  --tune-fast               快速 smoke mode，缩小搜索空间
  --retrain-decoupled       用 best_params.json 重训 decoupled 模型
  --use-decoupled           使用 decoupled 模型（而非 hybrid）生成提交包
  --decoupled-model-path    decoupled 模型路径
  --decoupled-best-params   调参最优参数 JSON（用于跳过调参直接重训）
  --decoupled-output-dir    decoupled 提交包解压目录
  --decoupled-zip-path      decoupled ZIP 输出路径

Single-Horizon H168 管道（辅助实验：不是当前正式提交格式）:
  --tune-single-horizon      运行 H168 单窗口模型超参数调优
  --retrain-single-horizon   用 best_params.json 重训 H168 模型
  --use-single-horizon       使用 H168 single-horizon 模型生成辅助包（不作为正式提交）
  --single-horizon-model-path H168 模型路径
  --single-horizon-best-params H168 调参最优参数 JSON
  --single-horizon-output-dir H168 提交包解压目录
  --single-horizon-zip-path  H168 ZIP 输出路径
EOF
}

resolve_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s\n' "${PROJECT_ROOT}/$1" ;;
    esac
}

pick_catalog() {
    if [ -n "$CATALOG" ]; then
        printf '%s\n' "$CATALOG"
    elif [ -f "data/raw/USGS_Mw4.0_Depth70_1970-2023.csv" ]; then
        printf '%s\n' "data/raw/USGS_Mw4.0_Depth70_1970-2023.csv"
    elif [ -f "data/raw/USGS_Mw4.5_Depth70_1970-2023.csv" ]; then
        printf '%s\n' "data/raw/USGS_Mw4.5_Depth70_1970-2023.csv"
    else
        printf '%s\n' "data/raw/USGS_Mw6.0_Depth70_1970-2023.csv"
    fi
}

pick_score_model() {
    if [ -n "$SCORE_MODEL" ]; then
        printf '%s\n' "$SCORE_MODEL"
    elif [ -f "data/models/qualification_best/qualification_window_models.joblib" ]; then
        printf '%s\n' "data/models/qualification_best/qualification_window_models.joblib"
    elif [ -f "data/models/qualification_window_models.joblib" ]; then
        printf '%s\n' "data/models/qualification_window_models.joblib"
    else
        printf '%s\n' "data/models/qualification_best/qualification_window_models.joblib"
    fi
}

sha256_print() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1"
    else
        shasum -a 256 "$1"
    fi
}

while [[ $# -gt 0 ]]; do
    arg="$1"
    case "$arg" in
        --input-dir) INPUT_DIR="$2"; shift ;;
        --input-dir=*) INPUT_DIR="${arg#*=}" ;;
        --features) FEATURE_DATA="$2"; shift ;;
        --features=*) FEATURE_DATA="${arg#*=}" ;;
        --base-features) BASE_FEATURES="$2"; shift ;;
        --base-features=*) BASE_FEATURES="${arg#*=}" ;;
        --catalog) CATALOG="$2"; shift ;;
        --catalog=*) CATALOG="${arg#*=}" ;;
        --score-model-path) SCORE_MODEL="$2"; shift ;;
        --score-model-path=*) SCORE_MODEL="${arg#*=}" ;;
        --legal-model-path) LEGAL_MODEL="$2"; shift ;;
        --legal-model-path=*) LEGAL_MODEL="${arg#*=}" ;;
        --output-dir) OUTPUT_DIR="$2"; shift ;;
        --output-dir=*) OUTPUT_DIR="${arg#*=}" ;;
        --zip-path) ZIP_PATH="$2"; shift ;;
        --zip-path=*) ZIP_PATH="${arg#*=}" ;;
        --commitment-template) COMMITMENT_TEMPLATE="$2"; SKIP_COMMITMENT=false; shift ;;
        --commitment-template=*) COMMITMENT_TEMPLATE="${arg#*=}"; SKIP_COMMITMENT=false ;;
        --with-commitment) COMMITMENT_TEMPLATE="$2"; SKIP_COMMITMENT=false; shift ;;
        --with-commitment=*) COMMITMENT_TEMPLATE="${arg#*=}"; SKIP_COMMITMENT=false ;;
        --skip-commitment) SKIP_COMMITMENT=true ;;
        --clean) CLEAN=true ;;
        --no-clean) CLEAN=false ;;
        --no-install) NO_INSTALL=true ;;
        --install) FORCE_INSTALL=true ;;
        --full-train) BUILD_LABELS=true; TRAIN_SCORE=true; TRAIN_LEGAL=true ;;
        --build-labels) BUILD_LABELS=true ;;
        --retrain-score) TRAIN_SCORE=true ;;
        --retrain-legal) TRAIN_LEGAL=true ;;
        --no-calibration) NO_CALIBRATION=true ;;
        --device) DEVICE="$2"; shift ;;
        --device=*) DEVICE="${arg#*=}" ;;
        --n-splits) N_SPLITS="$2"; shift ;;
        --n-splits=*) N_SPLITS="${arg#*=}" ;;
        --n-estimators) N_ESTIMATORS="$2"; shift ;;
        --n-estimators=*) N_ESTIMATORS="${arg#*=}" ;;
        --learning-rate) LEARNING_RATE="$2"; shift ;;
        --learning-rate=*) LEARNING_RATE="${arg#*=}" ;;
        --model-type) MODEL_TYPE="$2"; shift ;;
        --model-type=*) MODEL_TYPE="${arg#*=}" ;;
        --tune-decoupled) TUNE_DECOUPLED=true ;;
        --tune-trials) TUNE_TRIALS="$2"; shift ;;
        --tune-trials=*) TUNE_TRIALS="${arg#*=}" ;;
        --tune-objective) TUNE_OBJECTIVE="$2"; shift ;;
        --tune-objective=*) TUNE_OBJECTIVE="${arg#*=}" ;;
        --tune-fast) TUNE_FAST=true ;;
        --tune-decoupled-full) TUNE_DECOUPLED_FULL=true ;;
        --tune-target) TUNE_TARGET=""; shift ;;
        --tune-target=*) TUNE_TARGET="${arg#*=}" ;;
        --enable-oof-fusion) ENABLE_OOF_FUSION=true ;;
        --no-oof-fusion) ENABLE_OOF_FUSION=false ;;
        --tune-transformer) TUNE_TRANSFORMER=true ;;
        --tune-gnn) TUNE_GNN=true ;;
        --mag-trials) MAG_TRIALS="$2"; shift ;;
        --mag-trials=*) MAG_TRIALS="${arg#*=}" ;;
        --time-trials) TIME_TRIALS="$2"; shift ;;
        --time-trials=*) TIME_TRIALS="${arg#*=}" ;;
        --extreme-trials) EXTREME_TRIALS="$2"; shift ;;
        --extreme-trials=*) EXTREME_TRIALS="${arg#*=}" ;;
        --dl-trials) DL_TRIALS="$2"; shift ;;
        --dl-trials=*) DL_TRIALS="${arg#*=}" ;;
        --gnn-trials) GNN_TRIALS="$2"; shift ;;
        --gnn-trials=*) GNN_TRIALS="${arg#*=}" ;;
        --fusion-grid-step) FUSION_GRID_STEP="$2"; shift ;;
        --fusion-grid-step=*) FUSION_GRID_STEP="${arg#*=}" ;;
        --retrain-decoupled) RETRAIN_DECOUPLED=true ;;
        --use-decoupled) USE_DECOUPLED=true ;;
        --decoupled-model-path) DECOUPLED_MODEL_PATH="$2"; shift ;;
        --decoupled-model-path=*) DECOUPLED_MODEL_PATH="${arg#*=}" ;;
        --decoupled-best-params) DECOUPLED_BEST_PARAMS="$2"; shift ;;
        --decoupled-best-params=*) DECOUPLED_BEST_PARAMS="${arg#*=}" ;;
        --decoupled-output-dir) DECOUPLED_OUTPUT_DIR="$2"; shift ;;
        --decoupled-output-dir=*) DECOUPLED_OUTPUT_DIR="${arg#*=}" ;;
        --decoupled-zip-path) DECOUPLED_ZIP_PATH="$2"; shift ;;
        --decoupled-zip-path=*) DECOUPLED_ZIP_PATH="${arg#*=}" ;;
        # ── Single-horizon H168 ──
        --tune-single-horizon) TUNE_SINGLE_HORIZON=true ;;
        --retrain-single-horizon) RETRAIN_SINGLE_HORIZON=true ;;
        --use-single-horizon) USE_SINGLE_HORIZON=true ;;
        --single-horizon-model-path) SINGLE_HORIZON_MODEL_PATH="$2"; shift ;;
        --single-horizon-model-path=*) SINGLE_HORIZON_MODEL_PATH="${arg#*=}" ;;
        --single-horizon-best-params) SINGLE_HORIZON_BEST_PARAMS="$2"; shift ;;
        --single-horizon-best-params=*) SINGLE_HORIZON_BEST_PARAMS="${arg#*=}" ;;
        --single-horizon-output-dir) SINGLE_HORIZON_OUTPUT_DIR="$2"; shift ;;
        --single-horizon-output-dir=*) SINGLE_HORIZON_OUTPUT_DIR="${arg#*=}" ;;
        --single-horizon-zip-path) SINGLE_HORIZON_ZIP_PATH="$2"; shift ;;
        --single-horizon-zip-path=*) SINGLE_HORIZON_ZIP_PATH="${arg#*=}" ;;
        -h|--help) show_usage; exit 0 ;;
        qualification) ;;  # silently consume subcommand dispatch
        *) error "未知参数: $arg" ;;
    esac
    shift
done

SCORE_MODEL="$(pick_score_model)"
CATALOG="$(pick_catalog)"

# 默认 per-target trial 数回退到 TUNE_TRIALS
[ -z "$MAG_TRIALS" ] && MAG_TRIALS="$TUNE_TRIALS"
[ -z "$TIME_TRIALS" ] && TIME_TRIALS="$TUNE_TRIALS"
[ -z "$EXTREME_TRIALS" ] && EXTREME_TRIALS="$TUNE_TRIALS"

step "0. 环境检查"
"$PYTHON" --version
info "Python: $PYTHON"

step "1. 依赖检查"
if [ "$FORCE_INSTALL" = true ]; then
    info "强制安装依赖（跳过 lightgbm，避免覆盖 CUDA 构建）"
    grep -v '^lightgbm' requirements.txt | "$PYTHON" -m pip install -r /dev/stdin
elif [ "$NO_INSTALL" = true ]; then
    info "跳过依赖安装 (--no-install)"
else
    "$PYTHON" -c "import joblib, numpy, pandas, sklearn, xgboost; import lightgbm" 2>/dev/null && \
        info "依赖已就绪" || \
        { warn "依赖缺失，开始安装基础依赖（跳过 lightgbm）"; \
          grep -v '^lightgbm' requirements.txt | "$PYTHON" -m pip install -r /dev/stdin -q; }
fi

if [ "$BUILD_LABELS" = true ]; then
    step "2. 重建资格赛 T1/T2/T3 标签"
    [ -f "$CATALOG" ] || error "缺少完整地震目录: $CATALOG"
    "$PYTHON" main.py build-qualification-labels \
        --catalog "$CATALOG" \
        --base-features "$BASE_FEATURES" \
        --output "$FEATURE_DATA"
else
    info "跳过标签重建；使用: $FEATURE_DATA"
fi

if [ "$TRAIN_SCORE" = true ]; then
    step "3. 训练三窗口 score model"
    [ -f "$FEATURE_DATA" ] || error "缺少资格赛训练特征: $FEATURE_DATA"
    mkdir -p "$(dirname "$SCORE_MODEL")"
    "$PYTHON" main.py train-window-baseline \
        --data "$FEATURE_DATA" \
        --n-splits "$N_SPLITS" \
        --n-estimators "$N_ESTIMATORS" \
        --learning-rate "$LEARNING_RATE" \
        --model-type "$MODEL_TYPE" \
        --device "$DEVICE" \
        --use-asymmetric-time-objective \
        --save-dir "$(dirname "$SCORE_MODEL")"
else
    info "跳过 score model 重训；使用: $SCORE_MODEL"
fi

if [ "$TRAIN_LEGAL" = true ]; then
    step "4. 训练合法窗口 + 极端大余震风险模型"
    [ -f "$FEATURE_DATA" ] || error "缺少资格赛训练特征: $FEATURE_DATA"
    mkdir -p "$(dirname "$LEGAL_MODEL")"
    "$PYTHON" main.py train-legal-fusion \
        --data "$FEATURE_DATA" \
        --n-splits "$N_SPLITS" \
        --n-estimators "$N_ESTIMATORS" \
        --learning-rate "$LEARNING_RATE" \
        --model-type "$MODEL_TYPE" \
        --device "$DEVICE" \
        --use-asymmetric-time-objective \
        --save-dir "$(dirname "$LEGAL_MODEL")"
else
    info "跳过 legal-risk model 重训；使用: $LEGAL_MODEL"
fi

# ── Decoupled pipeline ──
if [ "$TUNE_DECOUPLED" = true ]; then
    step "D1. Decoupled 模型超参数调优"
    [ -f "$FEATURE_DATA" ] || error "缺少资格赛训练特征: $FEATURE_DATA"
    TUNE_OUTPUT="data/tuning_results/decoupled_$(date +%Y%m%d_%H%M%S)"
    TUNE_ARGS=(
        main.py tune-decoupled-models
        --data "$FEATURE_DATA"
        --n-trials "$TUNE_TRIALS"
        --n-splits "$N_SPLITS"
        --seed 42
        --device "$DEVICE"
        --objective "$TUNE_OBJECTIVE"
        --output-dir "$TUNE_OUTPUT"
    )
    if [ "$TUNE_FAST" = true ]; then
        TUNE_ARGS+=(--fast)
    fi
    "$PYTHON" "${TUNE_ARGS[@]}"
    DECOUPLED_BEST_PARAMS="${TUNE_OUTPUT}/best_params.json"
    info "调参结果: $DECOUPLED_BEST_PARAMS"
fi

if [ "$RETRAIN_DECOUPLED" = true ]; then
    step "D2. 用最优参数训练 decoupled 模型"
    [ -f "$FEATURE_DATA" ] || error "缺少资格赛训练特征: $FEATURE_DATA"
    mkdir -p "$(dirname "$DECOUPLED_MODEL_PATH")"
    D_TRAIN_ARGS=(
        main.py train-decoupled-window-models
        --data "$FEATURE_DATA"
        --n-splits "$N_SPLITS"
        --n-estimators "$N_ESTIMATORS"
        --learning-rate "$LEARNING_RATE"
        --model-type "$MODEL_TYPE"
        --device "$DEVICE"
        --save-dir "$(dirname "$DECOUPLED_MODEL_PATH")"
    )
    if [ -n "$DECOUPLED_BEST_PARAMS" ] && [ -f "$DECOUPLED_BEST_PARAMS" ]; then
        D_TRAIN_ARGS+=(--best-params "$DECOUPLED_BEST_PARAMS")
    fi
    "$PYTHON" "${D_TRAIN_ARGS[@]}"
    info "Decoupled 模型保存至: $(dirname "$DECOUPLED_MODEL_PATH")"
fi

# ── Single-Horizon H168 pipeline (formal route) ──
if [ "$TUNE_SINGLE_HORIZON" = true ]; then
    step "S1. H168 Single-Horizon 模型超参数调优"
    [ -f "$FEATURE_DATA" ] || error "缺少资格赛训练特征: $FEATURE_DATA"
    SH_TUNE_OUTPUT="data/tuning_results/single_horizon_$(date +%Y%m%d_%H%M%S)"
    SH_TUNE_ARGS=(
        main.py tune-single-horizon-models
        --data "$FEATURE_DATA"
        --mag-trials "$MAG_TRIALS"
        --time-trials "$TIME_TRIALS"
        --extreme-trials "$EXTREME_TRIALS"
        --n-splits "$N_SPLITS"
        --seed 42
        --device "$DEVICE"
        --output-dir "$SH_TUNE_OUTPUT"
    )
    if [ "$TUNE_FAST" = true ]; then
        SH_TUNE_ARGS+=(--fast)
    fi
    "$PYTHON" "${SH_TUNE_ARGS[@]}"
    SINGLE_HORIZON_BEST_PARAMS="${SH_TUNE_OUTPUT}/best_params.json"
    info "H168 调参结果: $SINGLE_HORIZON_BEST_PARAMS"
fi

if [ "$RETRAIN_SINGLE_HORIZON" = true ]; then
    step "S2. 用最优参数训练 H168 single-horizon 模型"
    [ -f "$FEATURE_DATA" ] || error "缺少资格赛训练特征: $FEATURE_DATA"
    mkdir -p "$(dirname "$SINGLE_HORIZON_MODEL_PATH")"
    SH_TRAIN_ARGS=(
        main.py train-single-horizon-models
        --data "$FEATURE_DATA"
        --n-splits "$N_SPLITS"
        --n-estimators "$N_ESTIMATORS"
        --learning-rate "$LEARNING_RATE"
        --model-type "$MODEL_TYPE"
        --device "$DEVICE"
        --save-dir "$(dirname "$SINGLE_HORIZON_MODEL_PATH")"
    )
    if [ -n "$SINGLE_HORIZON_BEST_PARAMS" ] && [ -f "$SINGLE_HORIZON_BEST_PARAMS" ]; then
        SH_TRAIN_ARGS+=(--best-params "$SINGLE_HORIZON_BEST_PARAMS")
    fi
    "$PYTHON" "${SH_TRAIN_ARGS[@]}"
    info "H168 模型保存至: $(dirname "$SINGLE_HORIZON_MODEL_PATH")"
fi

# ── Packaging ──
if [ "$USE_SINGLE_HORIZON" = true ]; then
    step "5. 生成 H168 single-horizon 资格赛提交包"
    [ -d "$INPUT_DIR" ] || error "缺少测试震例目录: $INPUT_DIR"
    [ -f "$SINGLE_HORIZON_MODEL_PATH" ] || error "缺少 H168 model: $SINGLE_HORIZON_MODEL_PATH。可加 --retrain-single-horizon 重训。"
    SH_PACKAGE_ARGS=(
        main.py make-single-horizon-package
        --input-dir "$INPUT_DIR"
        --model-path "$SINGLE_HORIZON_MODEL_PATH"
        --output-dir "$SINGLE_HORIZON_OUTPUT_DIR"
        --zip-path "$SINGLE_HORIZON_ZIP_PATH"
    )
    if [ "$CLEAN" = true ]; then
        SH_PACKAGE_ARGS+=(--clean)
    fi
    if [ "$SKIP_COMMITMENT" = true ]; then
        SH_PACKAGE_ARGS+=(--skip-commitment)
    else
        [ -f "$COMMITMENT_TEMPLATE" ] || error "承诺书模板不存在: $COMMITMENT_TEMPLATE"
        SH_PACKAGE_ARGS+=(--commitment-template "$COMMITMENT_TEMPLATE")
    fi
    "$PYTHON" "${SH_PACKAGE_ARGS[@]}"
    OUTPUT_DIR="$SINGLE_HORIZON_OUTPUT_DIR"
    ZIP_PATH="$SINGLE_HORIZON_ZIP_PATH"
elif [ "$USE_DECOUPLED" = true ]; then
    step "5. 生成 decoupled 资格赛提交包"
    [ -d "$INPUT_DIR" ] || error "缺少测试震例目录: $INPUT_DIR"
    [ -f "$DECOUPLED_MODEL_PATH" ] || error "缺少 decoupled model: $DECOUPLED_MODEL_PATH。可加 --retrain-decoupled 重训。"
    D_PACKAGE_ARGS=(
        main.py make-decoupled-qualification-package
        --input-dir "$INPUT_DIR"
        --model-path "$DECOUPLED_MODEL_PATH"
        --output-dir "$DECOUPLED_OUTPUT_DIR"
        --zip-path "$DECOUPLED_ZIP_PATH"
    )
    if [ "$CLEAN" = true ]; then
        D_PACKAGE_ARGS+=(--clean)
    fi
    if [ "$SKIP_COMMITMENT" = true ]; then
        D_PACKAGE_ARGS+=(--skip-commitment)
    else
        [ -f "$COMMITMENT_TEMPLATE" ] || error "承诺书模板不存在: $COMMITMENT_TEMPLATE"
        D_PACKAGE_ARGS+=(--commitment-template "$COMMITMENT_TEMPLATE")
    fi
    if [ "$NO_CALIBRATION" = true ]; then
        D_PACKAGE_ARGS+=(--mag-calibration none --time-calibration-hours none)
    fi
    "$PYTHON" "${D_PACKAGE_ARGS[@]}"
    # Update for validation
    OUTPUT_DIR="$DECOUPLED_OUTPUT_DIR"
    ZIP_PATH="$DECOUPLED_ZIP_PATH"
else
    step "5. 生成资格赛 hybrid 提交包"
[ -d "$INPUT_DIR" ] || error "缺少测试震例目录: $INPUT_DIR"
[ -f "$SCORE_MODEL" ] || error "缺少 score model: $SCORE_MODEL。可加 --retrain-score 或指定 --score-model-path。"
[ -f "$LEGAL_MODEL" ] || error "缺少 legal-risk model: $LEGAL_MODEL。可加 --retrain-legal 或指定 --legal-model-path。"

PACKAGE_ARGS=(
    main.py make-hybrid-qualification-package
    --input-dir "$INPUT_DIR"
    --score-model-path "$SCORE_MODEL"
    --legal-model-path "$LEGAL_MODEL"
    --output-dir "$OUTPUT_DIR"
    --zip-path "$ZIP_PATH"
)

if [ "$CLEAN" = true ]; then
    PACKAGE_ARGS+=(--clean)
fi
if [ "$SKIP_COMMITMENT" = true ]; then
    PACKAGE_ARGS+=(--skip-commitment)
else
    [ -f "$COMMITMENT_TEMPLATE" ] || error "承诺书模板不存在: $COMMITMENT_TEMPLATE"
    PACKAGE_ARGS+=(--commitment-template "$COMMITMENT_TEMPLATE")
fi
if [ "$NO_CALIBRATION" = true ]; then
    PACKAGE_ARGS+=(--mag-calibration none --time-calibration-hours none)
fi

"$PYTHON" "${PACKAGE_ARGS[@]}"
fi

step "6. 提交包结构校验"
ZIP_ABS="$(resolve_path "$ZIP_PATH")"
SKIP_COMMITMENT_ENV="$SKIP_COMMITMENT" \
  ZIP_PATH_ENV="$ZIP_ABS" \
  USE_SINGLE_HORIZON_ENV="$USE_SINGLE_HORIZON" \
  "$PYTHON" - <<'PY'
import os
import zipfile

zip_path = os.environ["ZIP_PATH_ENV"]
skip_commitment = os.environ.get("SKIP_COMMITMENT_ENV", "true") == "true"
use_single_horizon = os.environ.get("USE_SINGLE_HORIZON_ENV", "false") == "true"
with zipfile.ZipFile(zip_path) as zf:
    names = zf.namelist()
    csv_contents = {
        name: zf.read(name).decode("utf-8")
        for name in names
        if name.startswith("predictions/") and name.endswith(".csv")
    }
pred_files = [name for name in names if name.startswith("predictions/") and name.endswith(".csv")]
assert pred_files, "ZIP 中没有 predictions/*.csv"
if use_single_horizon:
    assert "predictions/qualification_predictions.csv" in names, (
        "H168 ZIP 中缺少 predictions/qualification_predictions.csv")
    assert not any(("T1-T2" in n or "-T3" in n) and n.startswith("predictions/") for n in names), (
        "H168 ZIP 不应包含 legacy T1-T2 / T3 文件")
else:
    t1t2_files = sorted(
        name for name in pred_files
        if name.endswith("-T1-T2.csv")
    )
    t3_files = sorted(
        name for name in pred_files
        if name.endswith("-T3.csv")
    )
    assert "predictions/qualification_predictions.csv" not in names, (
        "正式 T1/T2/T3 ZIP 不应包含 H168 qualification_predictions.csv")
    assert t1t2_files, "ZIP 中缺少 *-T1-T2.csv"
    assert t3_files, "ZIP 中缺少 *-T3.csv"
    assert len(t1t2_files) == len(t3_files), (
        f"T1-T2 文件数({len(t1t2_files)})与 T3 文件数({len(t3_files)})不一致")
    for name in t1t2_files:
        content = csv_contents[name].strip().splitlines()
        assert len(content) == 2, f"{name} 应为 2 行，对应 T1/T2"
    for name in t3_files:
        content = csv_contents[name].strip().splitlines()
        assert len(content) == 1, f"{name} 应为 1 行，对应 T3"
assert any(name.startswith("technical_docs/") for name in names), "ZIP 中没有 technical_docs/"
assert "MANIFEST.json" in names, "ZIP 中缺少 MANIFEST.json"
if not skip_commitment:
    assert any(name.startswith("commitment/") for name in names), "正式 ZIP 中缺少 commitment/"
print(f"ZIP 校验通过: predictions={len(pred_files)}, files={len(names)}")
PY

info "ZIP: $ZIP_ABS"
info "SHA256:"
sha256_print "$ZIP_ABS"

echo ""
echo "== 资格赛一键流程完成 =="
echo "  提交包目录: $(resolve_path "$OUTPUT_DIR")"
echo "  ZIP 文件:   $ZIP_ABS"
