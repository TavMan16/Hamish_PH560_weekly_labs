"""MPI driver for the Cython Task 3 Green's function runs."""

import argparse
import numpy as np
from mpi4py import MPI

from poisson_green_mpix import compute_green_function


def get_task3_sites(grid_size):
    """Return the three Task 3 start sites."""

    return {
        "centre": (grid_size // 2, grid_size // 2),
        "corner": (2, 2),
        "face": (grid_size // 2, 2),
    }


def save_result(result):
    """Save one Green's function result using the existing Cython naming style."""

    grid_size = result["grid_size"]
    start_row = result["start_row"]
    start_column = result["start_column"]
    total_walkers = result["total_walkers"]

    file_stub = f"N{grid_size}_r{start_row}_c{start_column}_w{total_walkers}"

    np.save(f"green_charge_{file_stub}x.npy", result["charge_green"])
    np.save(f"green_charge_std_{file_stub}x.npy", result["charge_std"])
    np.save(f"green_boundary_{file_stub}x.npy", result["boundary_green"])
    np.save(f"green_boundary_std_{file_stub}x.npy", result["boundary_std"])

    print(f"Saved green_charge_{file_stub}x.npy")
    print(f"Saved green_charge_std_{file_stub}x.npy")
    print(f"Saved green_boundary_{file_stub}x.npy")
    print(f"Saved green_boundary_std_{file_stub}x.npy")


def run_case(
    case_name,
    start_row,
    start_column,
    grid_size,
    grid_spacing,
    total_walkers,
    seed,
    comm,
):
    """Run one Task 3 case and save arrays on rank 0."""

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

    print()
    print(f"Finished case: {case_name}")
    print(f"Grid size: {result['grid_size']}")
    print(f"Start site: ({result['start_row']}, {result['start_column']})")
    print(f"Total walkers: {result['total_walkers']}")
    print(f"Parallel runtime: {parallel_runtime:.6f} s")

    save_result(result)


def main():
    """Configure and run one or all Task 3 Green's function calculations."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=["centre", "corner", "face", "all"],
        default="all",
        help="Task 3 start site to evaluate.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=101,
        help="Number of grid points in each direction.",
    )
    parser.add_argument(
        "--grid-spacing",
        type=float,
        default=0.01,
        help="Grid spacing in metres.",
    )
    parser.add_argument(
        "--walkers",
        type=int,
        default=1000000,
        help="Total number of random walkers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Base random seed.",
    )
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    task3_sites = get_task3_sites(args.grid_size)

    if args.case == "all":
        case_names = ["centre", "corner", "face"]
    else:
        case_names = [args.case]

    for case_name in case_names:
        start_row, start_column = task3_sites[case_name]
        run_case(
            case_name,
            start_row,
            start_column,
            args.grid_size,
            args.grid_spacing,
            args.walkers,
            args.seed,
            comm,
        )


if __name__ == "__main__":
    main()
