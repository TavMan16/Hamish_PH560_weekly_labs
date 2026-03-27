"""Plotting utilities for visualising the Task 1 Poisson solution."""

import matplotlib.pyplot as plt
import numpy as np

from poisson_sor_solver import build_boundary_array, solve_poisson


def create_test_source(grid_size):
    """Return a source distribution with a single charge at the grid centre."""

    source = np.zeros((grid_size, grid_size), dtype=float)
    centre = grid_size // 2
    source[centre, centre] = 1.0

    return source


def plot_heatmap(potential):
    """Display the potential field as a 2-D heatmap."""

    plt.figure(figsize=(6, 5))

    # imshow is the simplest way to inspect the full grid visually.
    image = plt.imshow(potential, origin="lower", cmap="coolwarm")
    plt.colorbar(image, label="Potential")
    plt.title("Poisson solution heatmap")
    plt.xlabel("x index")
    plt.ylabel("y index")

    plt.tight_layout()
    plt.show()


def plot_central_cross_section(potential):
    """Plot the potential along the central horizontal row."""

    grid_size = potential.shape[0]
    centre = grid_size // 2
    x_values = np.arange(grid_size)

    # A central slice is a quick way to inspect symmetry and decay.
    plt.figure(figsize=(6, 4))
    plt.plot(x_values, potential[centre, :], marker="o", markersize=3)
    plt.title("Central cross-section of the potential")
    plt.xlabel("x index")
    plt.ylabel("Potential")

    plt.tight_layout()
    plt.show()


def main():
    """Solve a simple test problem and plot the resulting potential."""

    grid_size = 50
    grid_spacing = 1.0
    tolerance = 1.0e-8
    max_iterations = 100000

    boundary_values = build_boundary_array(
        grid_size,
        0.0,
        0.0,
        2.0,
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

    plot_heatmap(result.potential)
    plot_central_cross_section(result.potential)


if __name__ == "__main__":
    main()
