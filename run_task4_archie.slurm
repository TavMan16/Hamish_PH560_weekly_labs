#!/bin/bash
#SBATCH --job-name=ph510_task4
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=ph510_task4_%j.out
#SBATCH --error=ph510_task4_%j.err

set -euo pipefail

echo "============================================================"
echo "PH510 Task 4 Wee Archie job"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "Tasks: ${SLURM_NTASKS}"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "============================================================"

# Load the modules used on your system.
# Replace these with the actual Wee Archie module names if needed.
module purge
module load python
module load openmpi

# Optional: show what environment was used in the report/log.
echo
echo "Loaded modules:"
module list 2>&1

echo
echo "Python executable:"
which python

echo
echo "Python version:"
python --version

# Build the Cython extension before running MPI.
echo
echo "============================================================"
echo "Building Cython extension"
echo "============================================================"
python setup.py build_ext --inplace

# Choose common parameters here.
GRID_SIZE=101
GRID_SPACING=0.01
WALKERS=1000000
SEED=12345

echo
echo "============================================================"
echo "Task parameters"
echo "============================================================"
echo "GRID_SIZE=${GRID_SIZE}"
echo "GRID_SPACING=${GRID_SPACING}"
echo "WALKERS=${WALKERS}"
echo "SEED=${SEED}"

# Measure total job-level wall clock for the MPI Green's-function stage.
GF_START=$(date +%s)

echo
echo "============================================================"
echo "Running MPI Green's-function evaluation"
echo "============================================================"

# Safest default: generate the full set needed downstream.
srun --ntasks=${SLURM_NTASKS} python run_task3x.py \
    --case all \
    --grid-size ${GRID_SIZE} \
    --grid-spacing ${GRID_SPACING} \
    --walkers ${WALKERS} \
    --seed ${SEED}

GF_END=$(date +%s)
GF_WALL=$((GF_END - GF_START))

echo
echo "============================================================"
echo "Green's-function wall clock time on 16 cores"
echo "============================================================"
echo "GF wall clock (seconds): ${GF_WALL}"
echo "GF wall clock (minutes): $(python - <<'PY'
gf_wall = int("${GF_WALL}")
print(f"{gf_wall / 60.0:.3f}")
PY
)"

# Post-processing is serial and should not use MPI.
echo
echo "============================================================"
echo "Running Task 3 plotting"
echo "============================================================"
python plot_task3x.py \
    --case all \
    --grid-size ${GRID_SIZE} \
    --grid-spacing ${GRID_SPACING} \
    --walkers ${WALKERS}

echo
echo "============================================================"
echo "Running Task 4 reconstruction"
echo "============================================================"
python reconstruct_potential.py \
    --case all \
    --grid-size ${GRID_SIZE} \
    --grid-spacing ${GRID_SPACING} \
    --walkers ${WALKERS}

echo
echo "============================================================"
echo "Running Task 4 plotting"
echo "============================================================"
python plot_task4x.py \
    --input task4_summary_N${GRID_SIZE}_w${WALKERS}x.csv \
    --grid-size ${GRID_SIZE} \
    --walkers ${WALKERS}

echo
echo "============================================================"
echo "Finished"
echo "End time: $(date)"
echo "============================================================"
