"""
Parallel midpoint-rule integration to estimate pi using MPI.

Each MPI process computes a subset of the integration points locally.
The partial results from all processes are summed together using MPI collective
communication to form the final estimate of pi.

The total number of integration samples is controlled via the environment
variable N_SAMPLES (set in the Slurm jobscript), allowing easy benchmarking
without modifying the Python code.

Tested with Python 3.12 and mpi4py.
"""

# Import standard library for accessing environment variables
import os

# Import the MPI interface from mpi4py
from mpi4py import MPI


def main() -> None:
    """
    Perform the parallel numerical integration using the midpoint rule.
    """

    # Get the MPI communicator that includes all running processes
    comm = MPI.COMM_WORLD

    # Get the rank (unique ID) of this MPI process
    # Ranks range from 0 to nproc - 1
    rank = comm.Get_rank()

    # Get the total number of MPI processes
    nproc = comm.Get_size()

    # Read total number of integration samples from environment variable
    # If N_SAMPLES is not set, default to 10,000,000
    n_samples = int(os.environ.get("N_SAMPLES", "10000000"))

    # Width of each small integration slice
    delta = 1.0 / n_samples

    # Synchronise all processes before starting the timer
    # (ensures fair timing of parallel computation)
    comm.Barrier()

    # Start timing the parallel computation
    start_time = MPI.Wtime()

    # Local accumulator for this process's partial sum
    local_sum = 0.0

    # Loop over the integration indices assigned to this rank
    #
    # Using a stride of nproc distributes the work evenly:
    #   rank 0 computes i = 0, nproc, 2*nproc, ...
    #   rank 1 computes i = 1, nproc+1, 2*nproc+1, ...
    #   etc.
    #
    for i in range(rank, n_samples, nproc):

        # Midpoint location for this slice
        x = (i + 0.5) * delta

        # Add this slice's contribution to the local sum
        local_sum += (4.0 / (1.0 + x * x)) * delta

    # Combine all local partial sums into a single total on rank 0
    #
    # MPI.SUM instructs MPI to add the values from all processes
    # root=0 means only rank 0 receives the final result
    total = comm.reduce(local_sum, op=MPI.SUM, root=0)

    # Stop timing on each process
    elapsed = MPI.Wtime() - start_time

    # Use an MPI reduction to find the maximum elapsed time across all ranks
    #
    # The slowest process determines the true parallel runtime
    elapsed_max = comm.allreduce(elapsed, op=MPI.MAX)

    # Only rank 0 prints output to avoid duplicated messages
    if rank == 0:
        # Optional: print the estimated value of pi
        # print(f"Estimated pi = {total:.12f}")

        # Print timing in a simple format for CSV parsing
        print(f"TIME {elapsed_max:.6f}")


# Standard Python entry point
# Ensures main() only runs when this file is executed directly
if __name__ == "__main__":
    main()

