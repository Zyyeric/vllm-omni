#!/bin/bash
#SBATCH --account=e32695
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=magihuman_cache_dit_container
#SBATCH --output=/scratch/cia5572/magihuman_cache_dit_bench/container_bench-%j.out
#SBATCH --error=/scratch/cia5572/magihuman_cache_dit_bench/container_bench-%j.err
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=mingyezhang2026@u.northwestern.edu

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_DIR="${REPO_DIR:-}"
if [[ -n "${REPO_DIR}" ]]; then
    :
elif [[ -f "${SUBMIT_DIR}/pyproject.toml" && -d "${SUBMIT_DIR}/vllm_omni" ]]; then
    REPO_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/../../pyproject.toml" && -d "${SUBMIT_DIR}/../../vllm_omni" ]]; then
    REPO_DIR="$(cd "${SUBMIT_DIR}/../.." && pwd)"
else
    REPO_DIR="/home/cia5572/vllm-omni"
fi

if [[ ! -f "${REPO_DIR}/examples/offline_inference/magi_human/end2end.py" ]]; then
    echo "Could not locate MagiHuman end2end.py. Set REPO_DIR=/path/to/vllm-omni when submitting." >&2
    echo "Tried REPO_DIR=${REPO_DIR} and SLURM_SUBMIT_DIR=${SUBMIT_DIR}" >&2
    exit 1
fi

module load singularity/3.8.1 || true

ENV_SETUP_SCRIPT="${ENV_SETUP_SCRIPT:-${HOME}/setup_vllm_omni_magihuman_container_env.sh}"
if [[ ! -f "${ENV_SETUP_SCRIPT}" ]]; then
    echo "Missing ${ENV_SETUP_SCRIPT}. Run setup_env_h100_apptainer.slurm first." >&2
    exit 1
fi
source "${ENV_SETUP_SCRIPT}"

MODEL="${MODEL:-SII-GAIR/daVinci-MagiHuman-Base-1080p}"
TP_SIZE="${TP_SIZE:-1}"
STEPS="${STEPS:-8}"
SEED="${SEED:-52}"
HEIGHT="${HEIGHT:-256}"
WIDTH="${WIDTH:-448}"
OUT_DIR="${OUT_DIR:-/scratch/cia5572/magihuman_cache_dit_bench/${SLURM_JOB_ID:-manual}}"
PROMPT="${PROMPT:-A young presenter with natural lip sync speaks warmly in a softly lit studio, with subtle camera motion and clear background ambience. Dialogue: \"Cache-DiT should make this MagiHuman generation faster while preserving the same output quality.\" Background Sound: <quiet studio ambience>}"

TMPDIR="${TMPDIR:-/scratch/cia5572/tmp}"
VLLM_RPC_BASE_PATH="${VLLM_RPC_BASE_PATH:-/scratch/cia5572/vllm_rpc}"

mkdir -p "${OUT_DIR}" "${HF_HOME}" "${HF_HUB_CACHE}" "${TMPDIR}" "${VLLM_RPC_BASE_PATH}"

run_in_container() {
    "${CONTAINER_RUNTIME:-apptainer}" exec --nv \
        --bind "${REPO_DIR}:${REPO_DIR}" \
        --bind "/scratch/cia5572:/scratch/cia5572" \
        --bind "${HOME}:${HOME}" \
        --env HF_HOME="${HF_HOME}" \
        --env HF_HUB_CACHE="${HF_HUB_CACHE}" \
        --env TMPDIR="${TMPDIR}" \
        --env VLLM_RPC_BASE_PATH="${VLLM_RPC_BASE_PATH}" \
        --env VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD}" \
        "${APPTAINER_IMAGE}" \
        bash -lc "source '${VLLM_OMNI_CONTAINER_ENV}/bin/activate' && cd '${REPO_DIR}/examples/offline_inference/magi_human' && $*"
}

run_in_container "python - <<'PY'
import cache_dit
import torch
import vllm
print(f'torch={torch.__version__}, cuda={torch.version.cuda}')
print(f'CUDA available: {torch.cuda.is_available()}, devices={torch.cuda.device_count()}')
print(f'vLLM={vllm.__version__}, cache_dit={cache_dit.__version__}')
PY"

echo "Baseline run: cache disabled"
run_in_container "python end2end.py \
    --model '${MODEL}' \
    --tensor-parallel-size '${TP_SIZE}' \
    --height '${HEIGHT}' \
    --width '${WIDTH}' \
    --num-inference-steps '${STEPS}' \
    --seed '${SEED}' \
    --prompt '${PROMPT}' \
    --output '${OUT_DIR}/magihuman_baseline.mp4'"

echo "Cache-DiT run: cache enabled"
run_in_container "python end2end.py \
    --model '${MODEL}' \
    --tensor-parallel-size '${TP_SIZE}' \
    --height '${HEIGHT}' \
    --width '${WIDTH}' \
    --num-inference-steps '${STEPS}' \
    --seed '${SEED}' \
    --prompt '${PROMPT}' \
    --cache-backend cache_dit \
    --enable-cache-dit-summary \
    --output '${OUT_DIR}/magihuman_cache_dit.mp4'"

echo "Benchmark complete. Compare the 'Generation time (generate only)' lines above."
