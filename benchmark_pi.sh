#!/bin/bash
#SBATCH --export=ALL
#SBATCH --partition=teaching
#SBATCH --account=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --job-name=pi_bench
#SBATCH --output=slurm-%j.out


# Clear any modules currently loaded in ARCHIE's environment
module purge

# Load the Python and MPI modules needed to run mpi4py jobs
module load python
module load mpi

# Set variable names and definitions used in the benchmark loop:
#   SCRIPT  = the Python program to run
#   OUT     = the output CSV file to write timings into
#   REPEATS = number of times to repeat each MPI run (for averaging later)
#   PCS     = list of MPI process counts (np values) to test
SCRIPT=pi_improved.py
OUT=timings.csv
REPEATS=5
PCS="1 2 4 8 16"

# Create/overwrite the CSV file and write a header row
# This makes the output easy to load later (e.g. in Python/pandas)
echo "np,run,time_seconds" > "$OUT"

# Loop over each requested number of MPI processes
for np in $PCS; do

  # For each process count, run the program REPEATS times
  for run in $(seq 1 $REPEATS); do

    # Run the program using srun with np MPI tasks.
    # The Python script is expected to print a line like:  TIME 0.123456
    # awk searches for a line starting with "TIME " and prints the 2nd column,
    # which is the numeric timing value.
    t=$(srun -n "$np" python3 "$SCRIPT" | awk '/^TIME /{print $2}')

    # Append one row to the CSV file: process_count, repeat_index, time_seconds
    echo "$np,$run,$t" >> "$OUT"

  done
done

# End of script:
# After the job finishes, timings.csv will contain all recorded timings,
# ready to be averaged and plotted.
