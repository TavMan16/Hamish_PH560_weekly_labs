"""MPI random-walk Green's function solver for the 2-D Poisson equation."""

import numpy as np
from mpi4py import MPI


def is_boundary_site(row_index, column_index, grid_size):
    """Return True if the site lies on the outer boundary of the grid."""

    return (
        row_index == 0
        or row_index == grid_size - 1
        or column_index == 0
        or column_index == grid_size - 1
    )


def split_walker_count(total_walkers, rank, size):
    """Split the total number of walkers as evenly as possible across ranks."""

    base_count = total_walkers // size
    remainder = total_walkers % size

    if rank < remainder:
        return base_count + 1

    return base_count


def take_random_step(row_index, column_index, rng):
    """Move one lattice step in a uniformly random nearest-neighbour direction."""

    step_direction = rng.integers(4)

    if step_direction == 0:
        return row_index + 1, column_index

    if step_direction == 1:
        return row_index - 1, column_index

    if step_direction == 2:
        return row_index, column_index + 1

    return row_index, column_index - 1


def run_single_walk(start_row, start_column, grid_size, rng):
    """
    Run one random walk until it first hits the boundary.

    The visit counter includes the starting site.
    """

    visit_counts = np.zeros((grid_size, grid_size), dtype=np.int64)
    exit_counts = np.zeros((grid_size, grid_size), dtype=np.int64)

    row_index = start_row
    column_index = start_column

    visit_counts[row_index, column_index] += 1

    while True:
        row_index, column_index = take_random_step(row_index, column_index, rng)

        if is_boundary_site(row_index, column_index, grid_size):
            exit_counts[row_index, column_index] = 1
            break

        visit_counts[row_index, column_index] += 1

    return visit_counts, exit_counts


def combine_mean_and_m2(count_a, mean_a, m2_a, count_b, mean_b, m2_b):
    """Combine two mean/M2 accumulators using the Chan formula."""

    if count_a == 0:
        return count_b, mean_b, m2_b

    if count_b == 0:
        return count_a, mean_a, m2_a

    delta = mean_b - mean_a
    total_count = count_a + count_b
    combined_mean = mean_a + delta * (count_b / total_count)
    combined_m2 = m2_a + m2_b + (delta * delta) * count_a * count_b / total_count

    return total_count, combined_mean, combined_m2


def accumulate_local_statistics(
    start_row,
    start_column,
    grid_size,
    local_walkers,
    grid_spacing,
    seed,
    rank,
):
    """Run the local walkers and accumulate mean/M2 fields for both Green's functions."""

    rng = np.random.default_rng(seed + rank)

    charge_mean = np.zeros((grid_size, grid_size), dtype=np.float64)
    charge_m2 = np.zeros((grid_size, grid_size), dtype=np.float64)

    boundary_mean = np.zeros((grid_size, grid_size), dtype=np.float64)
    boundary_m2 = np.zeros((grid_size, grid_size), dtype=np.float64)

    charge_count = 0
    boundary_count = 0

    spacing_squared = grid_spacing * grid_spacing

    for _ in range(local_walkers):
        visit_counts, exit_counts = run_single_walk(
            start_row,
            start_column,
            grid_size,
            rng,
        )

        # These are the per-walker contributions whose averages define the Green's functions.
        charge_sample = spacing_squared * visit_counts.astype(np.float64)
        boundary_sample = exit_counts.astype(np.float64)

        charge_count, charge_mean, charge_m2 = combine_mean_and_m2(
            charge_count,
            charge_mean,
            charge_m2,
            1,
            charge_sample,
            np.zeros_like(charge_sample),
        )

        boundary_count, boundary_mean, boundary_m2 = combine_mean_and_m2(
            boundary_count,
            boundary_mean,
            boundary_m2,
            1,
            boundary_sample,
            np.zeros_like(boundary_sample),
        )

    return {
        "charge_count": charge_count,
        "charge_mean": charge_mean,
        "charge_m2": charge_m2,
        "boundary_count": boundary_count,
        "boundary_mean": boundary_mean,
        "boundary_m2": boundary_m2,
    }


def reduce_statistics(local_statistics, comm):
    """Reduce local mean/M2 data onto rank 0."""

    gathered_statistics = comm.gather(local_statistics, root=0)

    if comm.rank != 0:
        return None

    first_item = gathered_statistics[0]
    grid_shape = first_item["charge_mean"].shape

    charge_count = 0
    charge_mean = np.zeros(grid_shape, dtype=np.float64)
    charge_m2 = np.zeros(grid_shape, dtype=np.float64)

    boundary_count = 0
    boundary_mean = np.zeros(grid_shape, dtype=np.float64)
    boundary_m2 = np.zeros(grid_shape, dtype=np.float64)

    for item in gathered_statistics:
        charge_count, charge_mean, charge_m2 = combine_mean_and_m2(
            charge_count,
            charge_mean,
            charge_m2,
            item["charge_count"],
            item["charge_mean"],
            item["charge_m2"],
        )

        boundary_count, boundary_mean, boundary_m2 = combine_mean_and_m2(
            boundary_count,
            boundary_mean,
            boundary_m2,
            item["boundary_count"],
            item["boundary_mean"],
            item["boundary_m2"],
        )

    return {
        "charge_count": charge_count,
        "charge_mean": charge_mean,
        "charge_m2": charge_m2,
        "boundary_count": boundary_count,
        "boundary_mean": boundary_mean,
        "boundary_m2": boundary_m2,
    }


def finalise_standard_deviation(sample_count, m2_array):
    """Return the sample standard deviation field."""

    if sample_count < 2:
        return np.zeros_like(m2_array)

    return np.sqrt(m2_array / (sample_count - 1))


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
    Estimate the Poisson Green's function from one start site using MPI-parallel random walks.

    Rank 0 returns the final result dictionary. Other ranks return None.
    """

    if is_boundary_site(start_row, start_column, grid_size):
        raise ValueError("The walker start site must lie inside the grid, not on the boundary.")

    local_walkers = split_walker_count(total_walkers, comm.rank, comm.size)

    local_statistics = accumulate_local_statistics(
        start_row,
        start_column,
        grid_size,
        local_walkers,
        grid_spacing,
        seed,
        comm.rank,
    )

    global_statistics = reduce_statistics(local_statistics, comm)

    if comm.rank != 0:
        return None

    charge_std = finalise_standard_deviation(
        global_statistics["charge_count"],
        global_statistics["charge_m2"],
    )

    boundary_std = finalise_standard_deviation(
        global_statistics["boundary_count"],
        global_statistics["boundary_m2"],
    )

    return {
        "charge_green": global_statistics["charge_mean"],
        "charge_std": charge_std,
        "boundary_green": global_statistics["boundary_mean"],
        "boundary_std": boundary_std,
        "total_walkers": global_statistics["charge_count"],
        "grid_size": grid_size,
        "grid_spacing": grid_spacing,
        "start_row": start_row,
        "start_column": start_column,
    }
