#!/bin/bash
#SBATCH --account=e32695
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=magihuman_cache_dit_bench
#SBATCH --output=/scratch/cia5572/magihuman_cache_dit_bench/slurm-%j.out
#SBATCH --error=/scratch/cia5572/magihuman_cache_dit_bench/slurm-%j.err
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=mingyezhang2026@u.northwestern.edu

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${SUBMIT_DIR}"
if [[ -f "end2end.py" ]]; then
    :
elif [[ -f "examples/offline_inference/magi_human/end2end.py" ]]; then
    cd "examples/offline_inference/magi_human"
else
    echo "Could not find MagiHuman end2end.py from ${SUBMIT_DIR}" >&2
    exit 1
fi

ENV_SETUP_SCRIPT="${ENV_SETUP_SCRIPT:-${HOME}/setup_vllm_omni_magihuman_env.sh}"
CONDA_ENV="${CONDA_ENV:-vllm-omni-magihuman}"
if [[ -f "${ENV_SETUP_SCRIPT}" ]]; then
    # Put cluster-specific module loads and environment activation in this file.
    source "${ENV_SETUP_SCRIPT}"
else
    if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    elif command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
    else
        echo "Conda is not available and ENV_SETUP_SCRIPT=${ENV_SETUP_SCRIPT} was not found." >&2
        exit 1
    fi
    conda activate "${CONDA_ENV}"
fi

python - <<'PY'
import cache_dit
import torch
import vllm

print(f"Python environment OK: torch={torch.__version__}, cuda={torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}, devices={torch.cuda.device_count()}")
print(f"vLLM={vllm.__version__}, cache_dit={cache_dit.__version__}")
PY

MODEL="${MODEL:-SII-GAIR/daVinci-MagiHuman-Base-1080p}"
TP_SIZE="${TP_SIZE:-1}"
STEPS="${STEPS:-8}"
SEED="${SEED:-52}"
HEIGHT="${HEIGHT:-256}"
WIDTH="${WIDTH:-448}"
OUT_DIR="${OUT_DIR:-/scratch/cia5572/magihuman_cache_dit_bench/${SLURM_JOB_ID:-manual}}"
PROMPT="${PROMPT:-A young presenter with natural lip sync speaks warmly in a softly lit studio, with subtle camera motion and clear background ambience. Dialogue: \"Cache-DiT should make this MagiHuman generation faster while preserving the same output quality.\" Background Sound: <quiet studio ambience>}"

mkdir -p "${OUT_DIR}"

echo "Running MagiHuman Cache-DiT benchmark"
echo "Model: ${MODEL}"
echo "Tensor parallel size: ${TP_SIZE}"
echo "Steps: ${STEPS}"
echo "Seed: ${SEED}"
echo "Output directory: ${OUT_DIR}"

echo "Baseline run: cache disabled"
python end2end.py \
    --model "${MODEL}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --num-inference-steps "${STEPS}" \
    --seed "${SEED}" \
    --prompt "${PROMPT}" \
    --output "${OUT_DIR}/magihuman_baseline.mp4"

echo "Cache-DiT run: cache enabled"
python end2end.py \
    --model "${MODEL}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --num-inference-steps "${STEPS}" \
    --seed "${SEED}" \
    --prompt "${PROMPT}" \
    --cache-backend cache_dit \
    --enable-cache-dit-summary \
    --output "${OUT_DIR}/magihuman_cache_dit.mp4"

echo "Benchmark complete. Compare the 'Generation time (generate only)' lines above."
