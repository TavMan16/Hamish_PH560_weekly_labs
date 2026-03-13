"""Minimal driver script for testing the Poisson SOR solver."""

import numpy as np

from poisson_sor_solver import build_boundary_array, solve_poisson


def create_test_source(grid_size):
    """Return a source distribution with a single charge at the grid centre."""

    source = np.zeros((grid_size, grid_size), dtype=float)

    # A central source in a grounded box gives a simple first test case.
    centre = grid_size // 2
    source[centre, centre] = 1.0

    return source


def main():
    """Configure and run a simple Poisson problem."""

    grid_size = 2000
    grid_spacing = 1.0
    tolerance = 1.0e-8
    max_iterations = 100000

    # Grounded boundaries keep the test case simple.
    boundary_values = build_boundary_array(
        grid_size,
        0.0,
        0.0,
        0.0,
        0.0,
    )

    source = create_test_source(grid_size)

    result = solve_poisson(
        grid_size,
        grid_spacing,
        source,
        boundary_values,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )

    print("Poisson solver finished")
    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    print(f"Final max update: {result.max_change:.3e}")
    print(f"Relaxation parameter omega: {result.omega:.4f}")

    centre = grid_size // 2
    print(f"Potential at centre: {result.potential[centre, centre]:.6f}")


if __name__ == "__main__":
    main()
