#!/bin/bash
#SBATCH --job-name=ph510_task4
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --partition=teaching
#SBATCH --account=teaching
#SBATCH --output=ph510_task4_%j.out
#SBATCH --error=ph510_task4_%j.err
#SBATCH --export=ALL

set -euo pipefail

echo "============================================================"
echo "PH510 Task 4 Wee Archie job"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "Tasks: ${SLURM_NTASKS}"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "============================================================"

echo
echo "============================================================"
echo "Loading modules"
echo "============================================================"
module purge
module load openmpi/gcc-14.2.1/4.1.8

echo
echo "Loaded modules:"
module list 2>&1

echo
echo "Python executable:"
which python3

echo
echo "Python version:"
python3 --version

echo
echo "============================================================"
echo "Checking Python packages"
echo "============================================================"
python3 -c "import Cython, numpy; print('Cython:', Cython.__version__); print('NumPy:', numpy.__version__)"

echo
echo "============================================================"
echo "Building Cython extension"
echo "============================================================"
python3 setup.py build_ext --inplace

GRID_SIZE=101
GRID_SPACING=0.01
WALKERS=10000000
SEED=12345

echo
echo "============================================================"
echo "Task parameters"
echo "============================================================"
echo "GRID_SIZE=${GRID_SIZE}"
echo "GRID_SPACING=${GRID_SPACING}"
echo "WALKERS=${WALKERS}"
echo "SEED=${SEED}"

GF_START=$(date +%s)

echo
echo "============================================================"
echo "Running MPI Green's-function evaluation"
echo "============================================================"

mpirun -np "${SLURM_NTASKS}" python3 run_task3x.py \
    --case all \
    --grid-size "${GRID_SIZE}" \
    --grid-spacing "${GRID_SPACING}" \
    --walkers "${WALKERS}" \
    --seed "${SEED}"

GF_END=$(date +%s)
GF_WALL=$((GF_END - GF_START))

echo
echo "============================================================"
echo "Green's-function wall clock time on ${SLURM_NTASKS} cores"
echo "============================================================"
echo "GF wall clock (seconds): ${GF_WALL}"
echo "GF wall clock (minutes): $(awk "BEGIN {printf \"%.3f\", ${GF_WALL}/60.0}")"

echo
echo "============================================================"
echo "Running Task 3 plotting"
echo "============================================================"
python3 plot_task3x.py \
    --case all \
    --grid-size "${GRID_SIZE}" \
    --grid-spacing "${GRID_SPACING}" \
    --walkers "${WALKERS}"

echo
echo "============================================================"
echo "Running Task 4 reconstruction"
echo "============================================================"
python3 reconstruct_potential.py \
    --case all \
    --grid-size "${GRID_SIZE}" \
    --grid-spacing "${GRID_SPACING}" \
    --walkers "${WALKERS}"

echo
echo "============================================================"
echo "Running Task 4 plotting"
echo "============================================================"
python3 plot_task4x.py \
    --input "task4_summary_N${GRID_SIZE}_w${WALKERS}x.csv" \
    --grid-size "${GRID_SIZE}" \
    --walkers "${WALKERS}"

echo
echo "============================================================"
echo "Finished"
echo "End time: $(date)"
echo "============================================================"
