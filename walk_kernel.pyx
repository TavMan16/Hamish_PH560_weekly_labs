# cython: boundscheck=False, wraparound=False, cdivision=True

"""Cython walk kernel for MPI Green's function accumulation."""

import numpy as np
cimport numpy as np

from libc.stdlib cimport rand, srand


ctypedef np.float64_t float64_t
ctypedef np.int32_t int32_t


cdef inline bint is_boundary_site(int row_index, int column_index, int grid_size):
    """Return True if the site lies on the outer boundary."""

    return (
        row_index == 0
        or row_index == grid_size - 1
        or column_index == 0
        or column_index == grid_size - 1
    )


cdef inline void take_random_step(int* row_index, int* column_index):
    """Take one uniformly random nearest-neighbour step."""

    cdef int direction = rand() % 4

    if direction == 0:
        row_index[0] += 1
    elif direction == 1:
        row_index[0] -= 1
    elif direction == 2:
        column_index[0] += 1
    else:
        column_index[0] -= 1


def accumulate_local_sums_cython(
    int start_row,
    int start_column,
    int grid_size,
    int local_walkers,
    double grid_spacing,
    unsigned int seed,
    int rank,
):
    """Accumulate local Green's function sums in compiled code."""

    cdef np.ndarray[float64_t, ndim=2] charge_sum
    cdef np.ndarray[float64_t, ndim=2] charge_sumsq
    cdef np.ndarray[float64_t, ndim=2] boundary_sum
    cdef np.ndarray[float64_t, ndim=2] boundary_sumsq
    cdef np.ndarray[int32_t, ndim=2] visit_counts

    cdef int walker_index
    cdef int row_index
    cdef int column_index
    cdef int exit_row
    cdef int exit_column
    cdef int row_loop
    cdef int column_loop

    cdef double spacing_squared = grid_spacing * grid_spacing
    cdef double sample_value

    charge_sum = np.zeros((grid_size, grid_size), dtype=np.float64)
    charge_sumsq = np.zeros((grid_size, grid_size), dtype=np.float64)
    boundary_sum = np.zeros((grid_size, grid_size), dtype=np.float64)
    boundary_sumsq = np.zeros((grid_size, grid_size), dtype=np.float64)

    srand(seed + rank)

    for walker_index in range(local_walkers):
        visit_counts = np.zeros((grid_size, grid_size), dtype=np.int32)

        row_index = start_row
        column_index = start_column
        visit_counts[row_index, column_index] += 1

        while True:
            take_random_step(&row_index, &column_index)

            if is_boundary_site(row_index, column_index, grid_size):
                exit_row = row_index
                exit_column = column_index
                break

            visit_counts[row_index, column_index] += 1

        for row_loop in range(grid_size):
            for column_loop in range(grid_size):
                if visit_counts[row_loop, column_loop] != 0:
                    sample_value = spacing_squared * visit_counts[row_loop, column_loop]
                    charge_sum[row_loop, column_loop] += sample_value
                    charge_sumsq[row_loop, column_loop] += sample_value * sample_value

        boundary_sum[exit_row, exit_column] += 1.0
        boundary_sumsq[exit_row, exit_column] += 1.0

    return charge_sum, charge_sumsq, boundary_sum, boundary_sumsq
