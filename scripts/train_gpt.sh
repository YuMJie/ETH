#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<USAGE
Usage: bash scripts/train_gpt.sh \\
  --config <config_json> \\
  --gpus-per-node <gpus_per_node> \\
  --num-nodes <num_nodes> \\
  --node-rank <node_rank> \\
  --master-addr <master_addr> \\
  --master-port <master_port> \\
  [--log-dir <runtime_log_dir>] \\
  [--train-iters <train_iters>] \\
  [--precision <fp32|fp16|bf16>] \\
  [--loss-scale <scale>] \\
  [--attention-dropout <probability>] \\
  [--hidden-dropout <probability>] \\
  [--clip-grad <max_norm>] \\
  [--grad-sync-mode <tensor|block>] \\
  [--grad-sync-bucket-size-mb <size_mb>] \\
  [--grad-sync-dump-plan <plan_json>] \\
  [--grad-sync-debug-replicated-check] \\
  [--grad-sync-metadata-transport <auto|object|tensor>] \\
  [--grad-sync-runtime-evidence-dir <runtime_evidence_dir>] \\
  [--evidence-dir <snapshot_dir>] \\
  [--evidence-canonical-init] \\
  [--vocab-padding-multiple <value>] \\
  [--python <python_path>] \\
  [--validate-only]
USAGE
}

CALLER_PWD="$PWD"

abs_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s/%s\n' "$CALLER_PWD" "$1" ;;
    esac
}

CONFIG=""
GPUS_PER_NODE=""
NUM_NODES=""
NODE_RANK=""
MASTER_ADDR=""
MASTER_PORT=""
LOG_DIR=""
TRAIN_ITERS=3
PRECISION="fp16"
LOSS_SCALE=""
ATTENTION_DROPOUT=""
HIDDEN_DROPOUT=""
CLIP_GRAD=""
GRAD_SYNC_MODE="tensor"
GRAD_SYNC_BUCKET_SIZE_MB=32
GRAD_SYNC_DUMP_PLAN=""
GRAD_SYNC_DEBUG_REPLICATED_CHECK=false
GRAD_SYNC_METADATA_TRANSPORT="auto"
GRAD_SYNC_RUNTIME_EVIDENCE_DIR=""
EVIDENCE_DIR=""
EVIDENCE_CANONICAL_INIT=false
VOCAB_PADDING_MULTIPLE=""
PYTHON_BIN="python3"
VALIDATE_ONLY=false

while [ "$#" -gt 0 ]; do
    case "$1" in
        --config)
            CONFIG="${2:-}"
            shift 2
            ;;
        --gpus-per-node)
            GPUS_PER_NODE="${2:-}"
            shift 2
            ;;
        --num-nodes)
            NUM_NODES="${2:-}"
            shift 2
            ;;
        --node-rank)
            NODE_RANK="${2:-}"
            shift 2
            ;;
        --master-addr)
            MASTER_ADDR="${2:-}"
            shift 2
            ;;
        --master-port)
            MASTER_PORT="${2:-}"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="${2:-}"
            shift 2
            ;;
        --train-iters)
            TRAIN_ITERS="${2:-}"
            shift 2
            ;;
        --precision)
            PRECISION="${2:-}"
            shift 2
            ;;
        --loss-scale)
            LOSS_SCALE="${2:-}"
            shift 2
            ;;
        --attention-dropout)
            ATTENTION_DROPOUT="${2:-}"
            shift 2
            ;;
        --hidden-dropout)
            HIDDEN_DROPOUT="${2:-}"
            shift 2
            ;;
        --clip-grad)
            CLIP_GRAD="${2:-}"
            shift 2
            ;;
        --grad-sync-mode)
            GRAD_SYNC_MODE="${2:-}"
            shift 2
            ;;
        --grad-sync-bucket-size-mb)
            GRAD_SYNC_BUCKET_SIZE_MB="${2:-}"
            shift 2
            ;;
        --grad-sync-dump-plan)
            GRAD_SYNC_DUMP_PLAN="${2:-}"
            shift 2
            ;;
        --grad-sync-debug-replicated-check)
            GRAD_SYNC_DEBUG_REPLICATED_CHECK=true
            shift
            ;;
        --grad-sync-metadata-transport)
            GRAD_SYNC_METADATA_TRANSPORT="${2:-}"
            shift 2
            ;;
        --grad-sync-runtime-evidence-dir)
            GRAD_SYNC_RUNTIME_EVIDENCE_DIR="${2:-}"
            shift 2
            ;;
        --evidence-dir)
            EVIDENCE_DIR="${2:-}"
            shift 2
            ;;
        --evidence-canonical-init)
            EVIDENCE_CANONICAL_INIT=true
            shift
            ;;
        --vocab-padding-multiple)
            VOCAB_PADDING_MULTIPLE="${2:-}"
            shift 2
            ;;
        --python)
            PYTHON_BIN="${2:-}"
            shift 2
            ;;
        --validate-only)
            VALIDATE_ONLY=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [ -z "${CONFIG}" ] || [ -z "${GPUS_PER_NODE}" ] || [ -z "${NUM_NODES}" ] || \
   [ -z "${NODE_RANK}" ] || [ -z "${MASTER_ADDR}" ] || [ -z "${MASTER_PORT}" ]; then
    usage
    exit 1
fi

case "${PRECISION}" in
    fp32|fp16|bf16) ;;
    *) echo "Unsupported precision: ${PRECISION}" >&2; exit 1 ;;
esac
case "${GRAD_SYNC_MODE}" in
    tensor|block) ;;
    *) echo "Unsupported grad sync mode: ${GRAD_SYNC_MODE}" >&2; exit 1 ;;
esac
case "${GRAD_SYNC_METADATA_TRANSPORT}" in
    auto|object|tensor) ;;
    *) echo "Unsupported grad sync metadata transport: ${GRAD_SYNC_METADATA_TRANSPORT}" >&2; exit 1 ;;
esac

for value_name in GPUS_PER_NODE NUM_NODES NODE_RANK MASTER_PORT TRAIN_ITERS; do
    value="${!value_name}"
    [[ "${value}" =~ ^[0-9]+$ ]] || { echo "${value_name} must be an integer" >&2; exit 1; }
done
[ "${GPUS_PER_NODE}" -gt 0 ] || { echo "GPUS_PER_NODE must be positive" >&2; exit 1; }
[ "${NUM_NODES}" -gt 0 ] || { echo "NUM_NODES must be positive" >&2; exit 1; }
[ "${NODE_RANK}" -lt "${NUM_NODES}" ] || { echo "NODE_RANK must be smaller than NUM_NODES" >&2; exit 1; }
[ "${MASTER_PORT}" -gt 0 ] && [ "${MASTER_PORT}" -le 65535 ] || { echo "MASTER_PORT must be between 1 and 65535" >&2; exit 1; }
[ "${TRAIN_ITERS}" -gt 0 ] || { echo "TRAIN_ITERS must be positive" >&2; exit 1; }
[[ "${GRAD_SYNC_BUCKET_SIZE_MB}" =~ ^[0-9]+$ ]] && [ "${GRAD_SYNC_BUCKET_SIZE_MB}" -gt 0 ] || \
    { echo "GRAD_SYNC_BUCKET_SIZE_MB must be a positive integer" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="$(abs_path "${CONFIG}")"
export PYTHONPATH="${REPO_ROOT}/eth/runtime:${REPO_ROOT}:${REPO_ROOT}/external/apex${PYTHONPATH:+:${PYTHONPATH}}"

if [ ! -f "${CONFIG_PATH}" ]; then
    echo "Config file not found: ${CONFIG_PATH}" >&2
    exit 1
fi

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || { echo "Python interpreter not found: ${PYTHON_BIN}" >&2; exit 1; }
WORLD_SIZE=$((GPUS_PER_NODE * NUM_NODES))
"${PYTHON_BIN}" "${REPO_ROOT}/tools/validation/validate_train_config.py" \
    --config "${CONFIG_PATH}" \
    --world-size "${WORLD_SIZE}" \
    --grad-sync-mode "${GRAD_SYNC_MODE}"

APEX_COMMIT="a651e2c24ecf97cbf367fd3f330df36760e1c597"
if [ ! -f "${REPO_ROOT}/external/apex/setup.py" ]; then
    echo "Apex submodule is missing; run git submodule update --init --recursive" >&2
    exit 1
fi
ACTUAL_APEX_COMMIT="$(git -C "${REPO_ROOT}/external/apex" rev-parse HEAD 2>/dev/null || true)"
if [ "${ACTUAL_APEX_COMMIT}" != "${APEX_COMMIT}" ]; then
    echo "Apex revision mismatch: expected ${APEX_COMMIT}, got ${ACTUAL_APEX_COMMIT:-unknown}" >&2
    exit 1
fi

if [ "${VALIDATE_ONLY}" = true ]; then
    echo "Training inputs validated; distributed launch skipped."
    exit 0
fi

if [ -z "${LOG_DIR}" ]; then
    CURRENT_TIME="$(date '+%Y-%m-%d-%H-%M-%S')"
    LOG_DIR="${REPO_ROOT}/logs-large/eth_gpt_runtime_${CURRENT_TIME}"
fi
LOG_DIR="$(abs_path "${LOG_DIR}")"
mkdir -p "${LOG_DIR}/csv"

mkdir -p "${LOG_DIR}/toolchain-bin"
PYTHON_COMMAND="$(command -v "${PYTHON_BIN}")"
PYTHON_BIN_DIR="$(cd "$(dirname "${PYTHON_COMMAND}")" && pwd)"
PYTHON_SCRIPTS_DIR="$("${PYTHON_COMMAND}" -c 'import sysconfig; print(sysconfig.get_path("scripts"))')"
ln -sf "${PYTHON_COMMAND}" "${LOG_DIR}/toolchain-bin/python"
export PATH="${LOG_DIR}/toolchain-bin:${PYTHON_SCRIPTS_DIR}:${PYTHON_BIN_DIR}:${PATH}"

if [ -n "${CUDA_HOME:-}" ]; then
    NVCC_BIN="${CUDA_HOME}/bin/nvcc"
elif command -v nvcc >/dev/null 2>&1; then
    NVCC_BIN="$(command -v nvcc)"
elif [ -x /usr/local/cuda/bin/nvcc ]; then
    NVCC_BIN="/usr/local/cuda/bin/nvcc"
else
    echo "nvcc not found; set CUDA_HOME or add nvcc to PATH" >&2
    exit 1
fi
if [ ! -x "${NVCC_BIN}" ]; then
    echo "nvcc not found or not executable: ${NVCC_BIN}" >&2
    exit 1
fi
export CUDA_HOME="$(cd "$(dirname "${NVCC_BIN}")/.." && pwd)"
"${PYTHON_BIN}" \
    "${REPO_ROOT}/tools/validation/validate_cuda_toolchain.py" \
    --nvcc "${NVCC_BIN}"

echo "Launching GPT training"
echo "  config: ${CONFIG_PATH}"
echo "  distributed: ${NUM_NODES} node(s), ${GPUS_PER_NODE} GPU(s)/node, rank ${NODE_RANK}"
echo "  master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  log_dir: ${LOG_DIR}"
echo "  train_iters: ${TRAIN_ITERS}"
echo "  precision: ${PRECISION}"
echo "  grad_sync: ${GRAD_SYNC_MODE}, ${GRAD_SYNC_BUCKET_SIZE_MB} MiB, ${GRAD_SYNC_METADATA_TRANSPORT} metadata"

cd "${REPO_ROOT}/eth/runtime"

TRAIN_ARGS=(
    pretrain_gpt.py
    --flexpipe-config "${CONFIG_PATH}"
    --train-iters "${TRAIN_ITERS}"
    --eval-iters 0
    --lr-decay-iters 320000
    --vocab-file vocabs/gpt2-vocab.json
    --merge-file vocabs/gpt2-merges.txt
    --data-impl mmap
    --split 949,50,1
    --distributed-backend nccl
    --lr 0.00015
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --weight-decay 1e-2
    --clip-grad 1.0
    --lr-warmup-fraction .01
    --log-interval 1
    --DDP-impl local
    --grad-sync-mode "${GRAD_SYNC_MODE}"
    --grad-sync-bucket-size-mb "${GRAD_SYNC_BUCKET_SIZE_MB}"
    --grad-sync-metadata-transport "${GRAD_SYNC_METADATA_TRANSPORT}"
    --log-path "${LOG_DIR%/}/"
)

case "${PRECISION}" in
    fp16) TRAIN_ARGS+=(--fp16) ;;
    bf16) TRAIN_ARGS+=(--bf16) ;;
esac
if [ -n "${LOSS_SCALE}" ]; then
    TRAIN_ARGS+=(--loss-scale "${LOSS_SCALE}")
fi
if [ -n "${ATTENTION_DROPOUT}" ]; then
    TRAIN_ARGS+=(--attention-dropout "${ATTENTION_DROPOUT}")
fi
if [ -n "${HIDDEN_DROPOUT}" ]; then
    TRAIN_ARGS+=(--hidden-dropout "${HIDDEN_DROPOUT}")
fi
if [ -n "${CLIP_GRAD}" ]; then
    TRAIN_ARGS+=(--clip-grad "${CLIP_GRAD}")
fi
if [ -n "${GRAD_SYNC_DUMP_PLAN}" ]; then
    TRAIN_ARGS+=(--grad-sync-dump-plan "$(abs_path "${GRAD_SYNC_DUMP_PLAN}")")
fi
if [ "${GRAD_SYNC_DEBUG_REPLICATED_CHECK}" = true ]; then
    TRAIN_ARGS+=(--grad-sync-debug-replicated-check)
fi
if [ -n "${GRAD_SYNC_RUNTIME_EVIDENCE_DIR}" ]; then
    TRAIN_ARGS+=(
        --grad-sync-runtime-evidence-dir
        "$(abs_path "${GRAD_SYNC_RUNTIME_EVIDENCE_DIR}")"
    )
fi
if [ -n "${EVIDENCE_DIR}" ]; then
    TRAIN_ARGS+=(--grad-sync-evidence-dir "$(abs_path "${EVIDENCE_DIR}")")
fi
if [ "${EVIDENCE_CANONICAL_INIT}" = true ]; then
    TRAIN_ARGS+=(--grad-sync-evidence-canonical-init)
fi
if [ -n "${VOCAB_PADDING_MULTIPLE}" ]; then
    TRAIN_ARGS+=(--make-vocab-size-divisible-by "${VOCAB_PADDING_MULTIPLE}")
fi

"${PYTHON_BIN}" -m torch.distributed.launch \
    --nproc_per_node "${GPUS_PER_NODE}" \
    --nnodes "${NUM_NODES}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${LOG_DIR}/gpt_train_rank${NODE_RANK}.log"
