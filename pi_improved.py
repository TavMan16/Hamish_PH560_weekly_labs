"""
Parallel midpoint-rule integration to estimate pi using MPI.

Each MPI process computes a subset of the integration points locally.
The parts of the answer from each process are then summed together using an MPI reduction.

Tested with Python 3.12.3 and mpi4py.
"""

# Import the MPI interface from mpi4py
from mpi4py import MPI


def integrand(x: float) -> float:
    """
    The function to be integrated.

    f(x) = 4 / (1 + x^2)

    The integral of this function from 0 to 1 is pi.
    """
    return 4.0 / (1.0 + x * x)


def main() -> None:
    """
    Perform the parallel numerical integration using the midpoint rule.
    """

    # Get the MPI communicator containing this process
    comm = MPI.COMM_WORLD

    # Asking MPI for the rank (ID) of each process
    # Ranks go from 0 to nproc - 1
    rank = comm.Get_rank()

    # Get the total number of MPI processes
    nproc = comm.Get_size()

    # Total number of integration samples
    n_samples = 10_000_000

    # Width of each small integration slice
    delta = 1.0 / n_samples

    # Start timing the parallel computation
    start_time = MPI.Wtime()

    # Local accumulator for this process's partial sum,
    # avoiding the original per-slice send/receive communication

    local_sum = 0.0

    # Loop over the integration indices assigned to this rank
    #
    # The range uses a stride of nproc so that:
    #   rank 0 computes i = 0, nproc, 2*nproc, ...
    #   rank 1 computes i = 1, nproc+1, 2*nproc+1, ...
    #   etc etc...
    #
    # This method should distribute the workload evenly across processes
    for i in range(rank, n_samples, nproc):

        # Midpoint location for this slice
        x = (i + 0.5) * delta

        # Add the area contribution for this slice
        local_sum += integrand(x) * delta

    # Combine all local partial sums into a single total on rank 0
    #
    # MPI.SUM tells MPI to add the values from all ranks together
    # root=0 means the final result is stored on rank 0 only
    total = comm.reduce(local_sum, op=MPI.SUM, root=0)

    # Stop the timer
    elapsed = MPI.Wtime() - start_time

    # Only rank 0 prints the final result and timing
    # (to avoid duplicated output from every process)
    if rank == 0:
        print(f"Estimated pi = {total:.12f}")
        print(f"Elapsed time (loop + reduce) = {elapsed:.6f} s")


# Standard Python pattern:
# Only run main() if this file is executed directly
if __name__ == "__main__":
    main()
