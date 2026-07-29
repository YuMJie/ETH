#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/search_gpt.sh [options]

Required:
  --profile-dir DIR          Completed output from scripts/profile_gpt.sh.
  --model-size SIZE          GPT model size.
  --search-budget SECONDS    Positive search time budget.
  --global-batch-size N      Positive global batch size.
  --node-description FILE    JSON containing node_info and device_info.
  --output DIR               Search logs, compatibility view, and generated config.

Optional:
  --python PATH              Python interpreter (default: python3).
  --dry-run                  Validate inputs and write the search plan without running search.
  -h, --help                 Show this help.
EOF
}

fail() {
    echo "search_gpt.sh: $*" >&2
    exit 2
}

require_value() {
    [ "$#" -ge 2 ] && [ -n "${2:-}" ] || fail "$1 requires a value"
}

PROFILE_DIR=""
MODEL_SIZE=""
SEARCH_BUDGET=""
GLOBAL_BATCH_SIZE=""
NODE_DESCRIPTION=""
OUTPUT_DIR=""
PYTHON_BIN="python3"
DRY_RUN=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --profile-dir|--model-size|--search-budget|--global-batch-size|--node-description|--output|--python)
            require_value "$@"
            case "$1" in
                --profile-dir) PROFILE_DIR="$2" ;;
                --model-size) MODEL_SIZE="$2" ;;
                --search-budget) SEARCH_BUDGET="$2" ;;
                --global-batch-size) GLOBAL_BATCH_SIZE="$2" ;;
                --node-description) NODE_DESCRIPTION="$2" ;;
                --output) OUTPUT_DIR="$2" ;;
                --python) PYTHON_BIN="$2" ;;
            esac
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            fail "unknown argument: $1"
            ;;
    esac
done

for required in PROFILE_DIR MODEL_SIZE SEARCH_BUDGET GLOBAL_BATCH_SIZE NODE_DESCRIPTION OUTPUT_DIR; do
    [ -n "${!required}" ] || { usage >&2; fail "missing required arguments"; }
done
case "$MODEL_SIZE" in
    350M|1_3B|2_6B|6_7B|13B) ;;
    *) fail "unsupported --model-size: $MODEL_SIZE" ;;
esac
[[ "$SEARCH_BUDGET" =~ ^[0-9]+$ ]] && [ "$SEARCH_BUDGET" -gt 0 ] || \
    fail "--search-budget must be a positive integer"
[[ "$GLOBAL_BATCH_SIZE" =~ ^[0-9]+$ ]] && [ "$GLOBAL_BATCH_SIZE" -gt 0 ] || \
    fail "--global-batch-size must be a positive integer"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "Python interpreter not found: $PYTHON_BIN"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PROFILE_DIR=$(cd "$PROFILE_DIR" 2>/dev/null && pwd) || fail "profile directory does not exist: $PROFILE_DIR"
NODE_DESCRIPTION=$(cd "$(dirname "$NODE_DESCRIPTION")" 2>/dev/null && pwd)/$(basename "$NODE_DESCRIPTION")

mapfile -t CONTRACT < <(
    "$PYTHON_BIN" -m eth.searching.profile_validation \
        --profile-dir "$PROFILE_DIR" \
        --model-size "$MODEL_SIZE" \
        --node-description "$NODE_DESCRIPTION" \
        --output "$OUTPUT_DIR"
)
[ "${#CONTRACT[@]}" -eq 4 ] || fail "profile validator returned an incomplete contract"
HARDWARE=${CONTRACT[0]}
MAX_TP=${CONTRACT[1]}
TOTAL_GPUS=${CONTRACT[2]}
NODE_INFO=${CONTRACT[3]}
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

PROFILE_VIEW="${OUTPUT_DIR}/profile_view"
mkdir -p "${PROFILE_VIEW}/${HARDWARE}"
ln -s "$PROFILE_DIR" "${PROFILE_VIEW}/${HARDWARE}/gpt_${MODEL_SIZE}"
ln -s "${PROFILE_DIR}/p2p_intra_node.csv" "${PROFILE_VIEW}/${HARDWARE}/p2p_intra_node.csv"
if [ -f "${PROFILE_DIR}/p2p_inter_node.csv" ]; then
    ln -s "${PROFILE_DIR}/p2p_inter_node.csv" "${PROFILE_VIEW}/p2p_inter_node.csv"
fi

START_STAGES=2
END_STAGES=$TOTAL_GPUS
[ "$END_STAGES" -le 8 ] || END_STAGES=8
[ "$END_STAGES" -ge "$START_STAGES" ] || fail "search requires at least two GPUs"
LOG_DIR="${OUTPUT_DIR}/logs"
CONFIG_DIR="${OUTPUT_DIR}/configs"
mkdir -p "$LOG_DIR" "${CONFIG_DIR}/top_configs" "${CONFIG_DIR}/csv"

cat > "${OUTPUT_DIR}/search_plan.txt" <<EOF
$PYTHON_BIN -m eth.searching.search --model-name gpt --model-size $MODEL_SIZE --global-batch-size $GLOBAL_BATCH_SIZE --node-info '$NODE_INFO' --device-json-path <output>/device_info.json --profiled-time-path <output>/profile_view/ --config-save-path <output>/configs/ --max-tp $MAX_TP --time-budget-total $SEARCH_BUDGET
EOF

if [ "$DRY_RUN" -eq 1 ]; then
    echo "Validated search plan: ${OUTPUT_DIR}/search_plan.txt"
    exit 0
fi

TIMESTAMP=$(date -u '+%Y-%m-%d-%H-%M-%S')
SAVE_PREFIX="eth_gpt_${MODEL_SIZE}_budget${SEARCH_BUDGET}"
cd "${REPO_ROOT}"
"$PYTHON_BIN" -m eth.searching.search \
    --model-name gpt \
    --model-size "$MODEL_SIZE" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --micro-batch-size 1 2 4 8 \
    --node-info "$NODE_INFO" \
    --device-json-path "${OUTPUT_DIR}/device_info.json" \
    --log-path "${LOG_DIR}/" \
    --profiled-time-path "${PROFILE_VIEW}/" \
    --config-save-path "${CONFIG_DIR}/" \
    --config-suffix "$TIMESTAMP" \
    --max-num-hops 7 \
    --time-budget-total "$SEARCH_BUDGET" \
    --initial-point balance_v1 \
    --num-of-saved-configs 1 \
    --support-comm-predict \
    --comm-revised \
    --op-grain layer \
    --save-prefix "$SAVE_PREFIX" \
    --max-tp "$MAX_TP" \
    --start-num-stages "$START_STAGES" \
    --end-num-stages "$END_STAGES" \
    --min-num-pipeline 1 \
    --max-num-pipeline 100 \
    --reduce-output \
    --add-action-tp-dp-exchange \
    --no-add-action-tp-dp \
    --use-cache \
    --optimal-prune \
    --sort-pipeline \
    --dp-find-pipeline \
    --homo-stage \
    --elastic-time \
    2>&1 | tee "${LOG_DIR}/search.log"

mapfile -t GENERATED_CONFIGS < <(find "${CONFIG_DIR}/top_configs" -maxdepth 1 -type f -name '*.json' -print)
[ "${#GENERATED_CONFIGS[@]}" -eq 1 ] || \
    fail "search must produce exactly one top config, found ${#GENERATED_CONFIGS[@]}"
"$PYTHON_BIN" -m eth.searching.config_adapter \
    --input "${GENERATED_CONFIGS[0]}" \
    --output "${OUTPUT_DIR}/gpt_config.json"
echo "Generated GPT config: ${OUTPUT_DIR}/gpt_config.json"
