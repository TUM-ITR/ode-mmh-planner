#!/bin/bash
#SBATCH --job-name=mc_ode_mmh_sim
#SBATCH --output=experiments/montecarlo/logs/out_%A_%a.out
#SBATCH --error=experiments/montecarlo/logs/err_%A_%a.err
#SBATCH --array=1-100%50        # Run seeds 1..100, max 50 in parallel
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2       # BLAS usually benefits from multiple threads
#SBATCH --mem=32G

# Load environment modules
# . /etc/profile.d/modules.sh
# module load apptainer

# Set variables
SEED=${SLURM_ARRAY_TASK_ID}
HOST_PROJECT_DIR="/sq/home/ge82sem/ifac2026"
CONTAINER_PROJECT_DIR="/home/developer/workspace"

# Create results directory paths
RUN_ID="job_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
HOST_RESULTS_DIR="${HOST_PROJECT_DIR}/experiments/montecarlo/results/${RUN_ID}"
CONTAINER_RESULTS_DIR="${CONTAINER_PROJECT_DIR}/experiments/montecarlo/results/${RUN_ID}"

mkdir -p "$HOST_RESULTS_DIR"

# Run inside Apptainer container
apptainer exec \
    --cleanenv \
    --bind ${HOST_PROJECT_DIR}:${CONTAINER_PROJECT_DIR} \
    --bind ${HOST_PROJECT_DIR}/.julia:/home/developer/.julia \
    ${HOST_PROJECT_DIR}/mc_ode_mmh_container.sif \
    julia --project=${CONTAINER_PROJECT_DIR} \
        ${CONTAINER_PROJECT_DIR}/experiments/montecarlo/run_montecarlo_single.jl \
        --seed ${SEED} \
        --results-dir ${CONTAINER_RESULTS_DIR}
