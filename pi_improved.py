"""
Parallel midpoint-rule integration to estimate pi using MPI.

Tested with Python 3.12.3 and mpi4py.
"""

from mpi4py import MPI


def integrand(x: float) -> float:
    """Function: 4/(1 + x^2)."""
    return 4.0 / (1.0 + x * x)


def main() -> None:
    """Computing the integral in parallel."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nproc = comm.Get_size()
    n_samples = 10_000_000
    delta = 1.0 / n_samples

    local_sum = 0.0

    for i in range(rank, n_samples, nproc):
        x = (i + 0.5) * delta
        local_sum += integrand(x) * delta

    total = comm.reduce(local_sum, op=MPI.SUM, root=0)

    if rank == 0:
        print(f"Estimated pi = {total:.12f}")


if __name__ == "__main__":
    main()
