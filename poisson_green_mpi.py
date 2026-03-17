"""MPI random-walk Green's function solver for the 2-D Poisson equation."""

import numpy as np
from mpi4py import MPI


def is_boundary_site(row_index, column_index, grid_size):
    """Return True if the site lies on the outer boundary."""

    return (
        row_index == 0
        or row_index == grid_size - 1
        or column_index == 0
        or column_index == grid_size - 1
    )


def split_walker_count(total_walkers, rank, size):
    """Split the walkers as evenly as possible across ranks."""

    base_count = total_walkers // size
    remainder = total_walkers % size

    if rank < remainder:
        return base_count + 1

    return base_count


def take_random_step(row_index, column_index, rng):
    """Take one uniformly random nearest-neighbour step."""

    direction = rng.integers(4)

    if direction == 0:
        return row_index + 1, column_index

    if direction == 1:
        return row_index - 1, column_index

    if direction == 2:
        return row_index, column_index + 1

    return row_index, column_index - 1


def run_single_walk(start_row, start_column, grid_size, rng):
    """
    Run one walker until it first reaches the boundary.

    The starting site is counted as one visit.
    """

    visit_counts = np.zeros((grid_size, grid_size), dtype=np.int32)

    row_index = start_row
    column_index = start_column

    visit_counts[row_index, column_index] += 1

    while True:
        row_index, column_index = take_random_step(row_index, column_index, rng)

        if is_boundary_site(row_index, column_index, grid_size):
            return visit_counts, row_index, column_index

        visit_counts[row_index, column_index] += 1


def accumulate_local_sums(
    start_row,
    start_column,
    grid_size,
    local_walkers,
    grid_spacing,
    seed,
    rank,
):
    """Accumulate local sums and sums of squares for the Green's functions."""

    rng = np.random.default_rng(seed + rank)
    spacing_squared = grid_spacing * grid_spacing

    charge_sum = np.zeros((grid_size, grid_size), dtype=np.float64)
    charge_sumsq = np.zeros((grid_size, grid_size), dtype=np.float64)

    boundary_sum = np.zeros((grid_size, grid_size), dtype=np.float64)
    boundary_sumsq = np.zeros((grid_size, grid_size), dtype=np.float64)

    for _ in range(local_walkers):
        visit_counts, exit_row, exit_column = run_single_walk(
            start_row,
            start_column,
            grid_size,
            rng,
        )

        # The charge Green's function is proportional to the visit count field.
        charge_sample = spacing_squared * visit_counts.astype(np.float64)
        charge_sum += charge_sample
        charge_sumsq += charge_sample * charge_sample

        # The boundary Green's function is an exit-location indicator field.
        boundary_sum[exit_row, exit_column] += 1.0
        boundary_sumsq[exit_row, exit_column] += 1.0

    return charge_sum, charge_sumsq, boundary_sum, boundary_sumsq


def compute_sample_std(sum_array, sumsq_array, sample_count):
    """Return the sample standard deviation field from sums and sums of squares."""

    if sample_count < 2:
        return np.zeros_like(sum_array)

    variance = (sumsq_array - (sum_array * sum_array) / sample_count) / (sample_count - 1)
    variance = np.maximum(variance, 0.0)

    return np.sqrt(variance)


def compute_green_function(
    start_row,
    start_column,
    grid_size,
    total_walkers,
    grid_spacing=1.0,
    seed=12345,
    comm=MPI.COMM_WORLD,
):
    """
    Estimate the Poisson Green's function from one start site using MPI random walks.

    Rank 0 returns the result dictionary. Other ranks return None.
    """

    if is_boundary_site(start_row, start_column, grid_size):
        raise ValueError("The walker start site must be inside the grid.")

    local_walkers = split_walker_count(total_walkers, comm.rank, comm.size)

    local_charge_sum, local_charge_sumsq, local_boundary_sum, local_boundary_sumsq = (
        accumulate_local_sums(
            start_row,
            start_column,
            grid_size,
            local_walkers,
            grid_spacing,
            seed,
            comm.rank,
        )
    )

    if comm.rank == 0:
        global_charge_sum = np.zeros_like(local_charge_sum)
        global_charge_sumsq = np.zeros_like(local_charge_sumsq)
        global_boundary_sum = np.zeros_like(local_boundary_sum)
        global_boundary_sumsq = np.zeros_like(local_boundary_sumsq)
    else:
        global_charge_sum = None
        global_charge_sumsq = None
        global_boundary_sum = None
        global_boundary_sumsq = None

    # Collective reductions avoid serial rank-0 recombination of Python objects.
    comm.Reduce(local_charge_sum, global_charge_sum, op=MPI.SUM, root=0)
    comm.Reduce(local_charge_sumsq, global_charge_sumsq, op=MPI.SUM, root=0)
    comm.Reduce(local_boundary_sum, global_boundary_sum, op=MPI.SUM, root=0)
    comm.Reduce(local_boundary_sumsq, global_boundary_sumsq, op=MPI.SUM, root=0)

    total_samples = comm.reduce(local_walkers, op=MPI.SUM, root=0)

    if comm.rank != 0:
        return None

    charge_mean = global_charge_sum / total_samples
    boundary_mean = global_boundary_sum / total_samples

    charge_std = compute_sample_std(
        global_charge_sum,
        global_charge_sumsq,
        total_samples,
    )
    boundary_std = compute_sample_std(
        global_boundary_sum,
        global_boundary_sumsq,
        total_samples,
    )

    return {
        "charge_green": charge_mean,
        "charge_std": charge_std,
        "boundary_green": boundary_mean,
        "boundary_std": boundary_std,
        "total_walkers": total_samples,
        "grid_size": grid_size,
        "grid_spacing": grid_spacing,
        "start_row": start_row,
        "start_column": start_column,
    }
