"""Minimal MPI driver for the Cython Task 2 Green's function solver."""

import numpy as np
from mpi4py import MPI

from poisson_green_mpix import compute_green_function


def main():
    """Configure and run one Green's function calculation."""

    comm = MPI.COMM_WORLD

    grid_size = 101
    grid_spacing = 0.01
    total_walkers = 1000000
    seed = 696967

    start_row = grid_size // 2
    start_column = grid_size // 2

    start_time = MPI.Wtime()

    result = compute_green_function(
        start_row,
        start_column,
        grid_size,
        total_walkers,
        grid_spacing=grid_spacing,
        seed=seed,
        comm=comm,
    )

    end_time = MPI.Wtime()
    local_runtime = end_time - start_time
    parallel_runtime = comm.reduce(local_runtime, op=MPI.MAX, root=0)

    if comm.rank != 0:
        return

    print("Green's function calculation finished")
    print(f"Grid size: {result['grid_size']}")
    print(f"Start site: ({result['start_row']}, {result['start_column']})")
    print(f"Total walkers: {result['total_walkers']}")
    print(f"Parallel runtime: {parallel_runtime:.6f} s")

    file_stub = f"N{grid_size}_r{start_row}_c{start_column}_w{total_walkers}"

    np.save(f"green_charge_{file_stub}x.npy", result["charge_green"])
    np.save(f"green_charge_std_{file_stub}x.npy", result["charge_std"])
    np.save(f"green_boundary_{file_stub}x.npy", result["boundary_green"])
    np.save(f"green_boundary_std_{file_stub}x.npy", result["boundary_std"])

    print(f"Saved green_charge_{file_stub}x.npy")
    print(f"Saved green_charge_std_{file_stub}x.npy")
    print(f"Saved green_boundary_{file_stub}x.npy")
    print(f"Saved green_boundary_std_{file_stub}x.npy")


if __name__ == "__main__":
    main()
