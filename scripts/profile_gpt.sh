#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/profile_gpt.sh [options]

Required:
  --model gpt                 Supported model family.
  --model-size SIZE           GPT size from eth/profiler/model_configs.py.
  --hardware LABEL            Stable hardware label used by search.
  --tp-sizes LIST             Comma-separated TP sizes, for example 1,2,4.
  --num-nodes N               Number of participating nodes.
  --gpus-per-node N           Visible GPUs on each node.
  --node-rank N               Zero-based rank of this node.
  --master-addr HOST          Distributed rendezvous host or address.
  --master-port PORT          Distributed rendezvous port.
  --output DIR                Self-contained profile output directory.

Optional:
  --python PATH               Python interpreter (default: python3).
  --reference-profile FILE   Reference op cache required for multi-node task splitting.
  --dry-run                  Validate inputs and write metadata plus the command plan only.
  -h, --help                 Show this help.
EOF
}

fail() {
    echo "profile_gpt.sh: $*" >&2
    exit 2
}

require_value() {
    [ "$#" -ge 2 ] && [ -n "${2:-}" ] || fail "$1 requires a value"
}

MODEL=""
MODEL_SIZE=""
HARDWARE=""
TP_SIZES=""
NUM_NODES=""
GPUS_PER_NODE=""
NODE_RANK=""
MASTER_ADDR=""
MASTER_PORT=""
OUTPUT_DIR=""
PYTHON_BIN="python3"
REFERENCE_PROFILE=""
DRY_RUN=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --model|--model-size|--hardware|--tp-sizes|--num-nodes|--gpus-per-node|--node-rank|--master-addr|--master-port|--output|--python|--reference-profile)
            require_value "$@"
            case "$1" in
                --model) MODEL="$2" ;;
                --model-size) MODEL_SIZE="$2" ;;
                --hardware) HARDWARE="$2" ;;
                --tp-sizes) TP_SIZES="$2" ;;
                --num-nodes) NUM_NODES="$2" ;;
                --gpus-per-node) GPUS_PER_NODE="$2" ;;
                --node-rank) NODE_RANK="$2" ;;
                --master-addr) MASTER_ADDR="$2" ;;
                --master-port) MASTER_PORT="$2" ;;
                --output) OUTPUT_DIR="$2" ;;
                --python) PYTHON_BIN="$2" ;;
                --reference-profile) REFERENCE_PROFILE="$2" ;;
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

for required in MODEL MODEL_SIZE HARDWARE TP_SIZES NUM_NODES GPUS_PER_NODE NODE_RANK MASTER_ADDR MASTER_PORT OUTPUT_DIR; do
    [ -n "${!required}" ] || { usage >&2; fail "missing required arguments"; }
done

[ "$MODEL" = "gpt" ] || fail "--model must be gpt"
case "$MODEL_SIZE" in
    350M|1_3B|2_6B|6_7B|13B|scale-layer) ;;
    *) fail "unsupported --model-size: $MODEL_SIZE" ;;
esac

for value_name in NUM_NODES GPUS_PER_NODE NODE_RANK MASTER_PORT; do
    value="${!value_name}"
    [[ "$value" =~ ^[0-9]+$ ]] || fail "--${value_name,,} must be an integer"
done
[ "$NUM_NODES" -ge 1 ] || fail "--num-nodes must be positive"
[ "$GPUS_PER_NODE" -ge 1 ] || fail "--gpus-per-node must be positive"
[ "$NODE_RANK" -lt "$NUM_NODES" ] || fail "--node-rank must be smaller than --num-nodes"
[ "$MASTER_PORT" -ge 1 ] && [ "$MASTER_PORT" -le 65535 ] || fail "--master-port must be between 1 and 65535"

IFS=',' read -r -a TP_VALUES <<< "$TP_SIZES"
[ "${#TP_VALUES[@]}" -gt 0 ] || fail "--tp-sizes cannot be empty"
for tp in "${TP_VALUES[@]}"; do
    [[ "$tp" =~ ^[0-9]+$ ]] || fail "each TP size must be an integer"
    [ "$tp" -ge 1 ] || fail "each TP size must be positive"
    [ $((tp & (tp - 1))) -eq 0 ] || fail "each TP size must be a power of two"
    [ "$tp" -le "$GPUS_PER_NODE" ] || fail "TP size $tp exceeds --gpus-per-node $GPUS_PER_NODE"
done

command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "Python interpreter not found: $PYTHON_BIN"
[ "$DRY_RUN" -eq 1 ] || [ "$NUM_NODES" -eq 1 ] || [ -f "$REFERENCE_PROFILE" ] || \
    fail "--reference-profile is required for multi-node profiling"
if [ "$DRY_RUN" -eq 0 ]; then
    "$PYTHON_BIN" -c \
        'import sys, torch; expected = int(sys.argv[1]); sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() >= expected else 1)' \
        "$GPUS_PER_NODE" || fail "CUDA is unavailable or fewer than $GPUS_PER_NODE devices are visible"
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
export PYTHONPATH="${REPO_ROOT}/eth/runtime:${REPO_ROOT}:${REPO_ROOT}/external/apex${PYTHONPATH:+:${PYTHONPATH}}"
for path in eth/profiler/op_profiler.py eth/profiler/comm_profiler.py eth/profiler/p2p_band_profiler.py eth/profiler/profile_metadata.py; do
    [ -f "${REPO_ROOT}/${path}" ] || fail "required profiler component is missing: $path"
done

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
PLAN_FILE="${OUTPUT_DIR}/profile_plan.txt"

{
    for tp in "${TP_VALUES[@]}"; do
        echo "$PYTHON_BIN -m torch.distributed.launch --module --nproc_per_node $tp --nnodes $NUM_NODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT eth.profiler.op_profiler --prof-tp-size $tp --prof-path <output>/ --prof-cache-file <output>/gpt_op_profile.pkl --prof-model-name gpt --prof-model-size $MODEL_SIZE"
        echo "$PYTHON_BIN -m eth.profiler.comm_profiler --prof-path <output>/ --prof-cache-file <output>/gpt_comm_profile.pkl --prof-op-time-path <output>/ --prof-tp-size $tp --prof-model-name gpt --prof-model-size $MODEL_SIZE"
    done
    echo "$PYTHON_BIN -m eth.profiler.p2p_band_profiler # writes <output>/p2p_intra_node.csv or p2p_inter_node.csv"
} > "$PLAN_FILE"

write_metadata() {
    local status=$1
    "$PYTHON_BIN" -m eth.profiler.profile_metadata \
        --output "$OUTPUT_DIR" \
        --status "$status" \
        --model "$MODEL" \
        --model-size "$MODEL_SIZE" \
        --hardware "$HARDWARE" \
        --tp-sizes "$TP_SIZES" \
        --num-nodes "$NUM_NODES" \
        --gpus-per-node "$GPUS_PER_NODE" \
        --node-rank "$NODE_RANK"
}

if [ "$DRY_RUN" -eq 1 ]; then
    write_metadata planned
    echo "Validated profiling plan: $PLAN_FILE"
    exit 0
fi

cd "$REPO_ROOT"
for tp in "${TP_VALUES[@]}"; do
    OP_ARGS=(
        --prof-tp-size "$tp"
        --prof-path "${OUTPUT_DIR}/"
        --prof-cache-file "${OUTPUT_DIR}/gpt_op_profile.pkl"
        --prof-model-name gpt
        --prof-model-size "$MODEL_SIZE"
        --prof-warmup-times 10
        --prof-repeat-times 40
    )
    if [ "$NUM_NODES" -gt 1 ]; then
        OP_ARGS+=(
            --prof-num-nodes "$NUM_NODES"
            --prof-node-rank "$NODE_RANK"
            --prof-ref-data "$REFERENCE_PROFILE"
        )
    fi
    "$PYTHON_BIN" -m torch.distributed.launch \
        --module \
        --nproc_per_node "$tp" \
        --nnodes "$NUM_NODES" \
        --node_rank "$NODE_RANK" \
        --master_addr "$MASTER_ADDR" \
        --master_port "$MASTER_PORT" \
        eth.profiler.op_profiler "${OP_ARGS[@]}" \
        2>&1 | tee "${OUTPUT_DIR}/op_tp${tp}_rank${NODE_RANK}.log"

    if [ "$NODE_RANK" -eq 0 ]; then
        MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT" \
            "$PYTHON_BIN" -m eth.profiler.comm_profiler \
            --prof-path "${OUTPUT_DIR}/" \
            --prof-cache-file "${OUTPUT_DIR}/gpt_comm_profile.pkl" \
            --prof-op-time-path "${OUTPUT_DIR}/" \
            --prof-tp-size "$tp" \
            --prof-model-name gpt \
            --prof-model-size "$MODEL_SIZE" \
            --prof-warmup-times 5 \
            --prof-repeat-times 20 \
            --max-data-size 4096 \
            2>&1 | tee "${OUTPUT_DIR}/comm_tp${tp}.log"
    fi
done

if [ "$NODE_RANK" -eq 0 ] && [ "$GPUS_PER_NODE" -ge 2 ]; then
    MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT" NNODES=1 GPUS_PER_NODE=2 NODE_RANK=0 \
        FILE_NAME="${OUTPUT_DIR}/p2p_intra_node.csv" \
        "$PYTHON_BIN" -m eth.profiler.p2p_band_profiler \
        2>&1 | tee "${OUTPUT_DIR}/p2p_intra_node.log"
fi

if [ "$NUM_NODES" -eq 2 ]; then
    MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT" NNODES=2 GPUS_PER_NODE=1 NODE_RANK="$NODE_RANK" \
        FILE_NAME="${OUTPUT_DIR}/p2p_inter_node.csv" \
        "$PYTHON_BIN" -m eth.profiler.p2p_band_profiler \
        2>&1 | tee "${OUTPUT_DIR}/p2p_inter_node_rank${NODE_RANK}.log"
fi

write_metadata complete
echo "Profiling complete: $OUTPUT_DIR"
