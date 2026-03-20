#!/bin/bash
#SBATCH --job-name=task1
#SBATCH --partition=teaching
#SBATCH --account=teaching
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --output=task1_%j.out
#SBATCH --error=task1_%j.err

set -e

echo "Running Task 1 on compute node"

python3 run_task1.py
