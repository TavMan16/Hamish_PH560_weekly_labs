#!/bin/bash
# Use the bash shell to run this script


# Request 1 compute node
#SBATCH --nodes=1

# Request up to 16 MPI tasks (processes)
#SBATCH --ntasks=16

# Use the whole node exclusively
#SBATCH --exclusive

# Set maximum runtime to 20 minutes
#SBATCH --time=00:20:00

# Name of the job in the scheduler
#SBATCH --job-name=pi_benchmark

# File for standard output (%j is replaced with job ID)
#SBATCH --output=slurm-%j.out

# Choose the teaching partition (queue)
#SBATCH --partition=teaching

# Account to charge compute time to
#SBATCH --account=teaching


# Remove all currently loaded modules
module purge

# Load the MPI environment (needed for mpi4py)
module load mpi


# CSV file to store all timing results
OUT="timings_both.csv"

# Number of repeated runs per configuration
REPEATS=5

# MPI process counts to test
PCS="1 2 4 8 16"


# Python implementations to benchmark
SCRIPTS="assignment1.py pi_improved.py"


# Write the CSV header (overwrite if file exists)
echo "implementation,np,run,pi_estimate,time_seconds" > "$OUT"


# Loop over each Python script
for script in $SCRIPTS; do

  # Loop over each MPI process count
  for np in $PCS; do

    # Repeat each configuration several times
    for run in $(seq 1 $REPEATS); do


      # Run the Python script with np MPI processes and capture output
      output=$(srun --mpi=pmi2 -n "$np" python "$script")


      # Extract the pi estimate from the line beginning with "PI "
      pi_value=$(echo "$output" | awk '/^PI /{print $2}')

      # Extract the timing from the line beginning with "TIME "
      time_value=$(echo "$output" | awk '/^TIME /{print $2}')


      # Check that both values were successfully extracted
      if [[ -z "$pi_value" || -z "$time_value" ]]; then

        # Print an error message
        echo "Error running $script with np=$np run=$run"

        # Print full output for debugging
        echo "$output"

        # Stop execution if something failed
        exit 1

      fi


      # Append the results as a new row in the CSV file
      echo "$script,$np,$run,$pi_value,$time_value" >> "$OUT"


      # Print progress to the terminal
      echo "impl=$script np=$np run=$run time=$time_value"

    done
  done
done

