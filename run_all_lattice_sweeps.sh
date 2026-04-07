#!/bin/bash

# Run Ising and XY temperature sweeps for multiple lattice sizes.
# Each run writes its own CSV file and a terminal log.

# Number of MPI processes to use for every run.
NP=8

# Lattice sizes to test.
LATTICE_SIZES=(8 16 32 64)

# Python scripts to run.
ISING_SCRIPT="run_temperature_sweep.py"
XY_SCRIPT="run_xy_temperature_sweep.py"

# Folder for terminal logs.
LOG_DIR="overnight_logs"

# Create the log directory if it does not already exist.
mkdir -p "$LOG_DIR"

# Print a short header.
echo "Starting overnight lattice-size sweep runs"
echo "MPI processes: $NP"
echo "Lattice sizes: ${LATTICE_SIZES[*]}"
echo "Start time: $(date)"
echo

# Loop over all lattice sizes.
for L in "${LATTICE_SIZES[@]}"
do
    echo "========================================"
    echo "Running Ising sweep for L=$L"
    echo "Time: $(date)"
    echo "========================================"

    LENGTH=$L mpirun -np "$NP" python3 "$ISING_SCRIPT" \
        | tee "$LOG_DIR/ising_L${L}_np${NP}.log"

    echo
    echo "Finished Ising sweep for L=$L"
    echo "Time: $(date)"
    echo

    echo "========================================"
    echo "Running XY sweep for L=$L"
    echo "Time: $(date)"
    echo "========================================"

    LENGTH=$L mpirun -np "$NP" python3 "$XY_SCRIPT" \
        | tee "$LOG_DIR/xy_L${L}_np${NP}.log"

    echo
    echo "Finished XY sweep for L=$L"
    echo "Time: $(date)"
    echo
done

echo "All runs completed"
echo "Finish time: $(date)"
