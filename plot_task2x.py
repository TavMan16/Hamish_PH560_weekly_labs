"""Plot saved Green's function data from the Cython Task 2 solver."""

import matplotlib.pyplot as plt
import numpy as np


def plot_charge_heatmap(charge_green, grid_size, start_row, start_column, total_walkers):
    """Save a heatmap of the charge-related Green's function."""

    plt.figure(figsize=(6, 5))

    image = plt.imshow(charge_green, origin="lower", cmap="viridis")
    plt.colorbar(image, label="Charge Green's function")
    plt.title(
        "Charge Green's function "
        f"(N={grid_size}, start=({start_row}, {start_column}), walkers={total_walkers}, cython)"
    )
    plt.xlabel("Column index")
    plt.ylabel("Row index")

    plt.tight_layout()
    plt.savefig(
        f"green_charge_heatmap_N{grid_size}_r{start_row}_c{start_column}_w{total_walkers}x.png",
        dpi=300,
    )
    plt.close()


def plot_charge_surface(charge_green, grid_size, start_row, start_column, total_walkers):
    """Save a 3-D surface plot of the charge-related Green's function."""

    row_values = np.arange(grid_size)
    column_values = np.arange(grid_size)
    column_grid, row_grid = np.meshgrid(column_values, row_values)

    figure = plt.figure(figsize=(8, 6))
    axis = figure.add_subplot(111, projection="3d")

    surface = axis.plot_surface(
        column_grid,
        row_grid,
        charge_green,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
    )

    figure.colorbar(surface, ax=axis, shrink=0.7, label="Charge Green's function")
    axis.set_title(
        "Charge Green's function surface "
        f"(N={grid_size}, start=({start_row}, {start_column}), walkers={total_walkers}, cython)"
    )
    axis.set_xlabel("Column index")
    axis.set_ylabel("Row index")
    axis.set_zlabel("Value")

    plt.tight_layout()
    plt.savefig(
        f"green_charge_surface_N{grid_size}_r{start_row}_c{start_column}_w{total_walkers}x.png",
        dpi=300,
    )
    plt.close()


def plot_boundary_heatmap(boundary_green, grid_size, start_row, start_column, total_walkers):
    """Save a heatmap of the boundary-related Green's function."""

    plt.figure(figsize=(6, 5))

    image = plt.imshow(boundary_green, origin="lower", cmap="magma")
    plt.colorbar(image, label="Boundary Green's function")
    plt.title(
        "Boundary Green's function "
        f"(N={grid_size}, start=({start_row}, {start_column}), walkers={total_walkers}, cython)"
    )
    plt.xlabel("Column index")
    plt.ylabel("Row index")

    plt.tight_layout()
    plt.savefig(
        f"green_boundary_heatmap_N{grid_size}_r{start_row}_c{start_column}_w{total_walkers}x.png",
        dpi=300,
    )
    plt.close()


def main():
    """Load saved Cython Green's function arrays and generate plot files."""

    grid_size = 101
    start_row = grid_size // 2
    start_column = grid_size // 2
    total_walkers = 1000000

    file_stub = f"N{grid_size}_r{start_row}_c{start_column}_w{total_walkers}"

    charge_green = np.load(f"green_charge_{file_stub}x.npy")
    boundary_green = np.load(f"green_boundary_{file_stub}x.npy")

    print(f"Loaded green_charge_{file_stub}x.npy")
    print(f"Loaded green_boundary_{file_stub}x.npy")

    plot_charge_heatmap(
        charge_green,
        grid_size,
        start_row,
        start_column,
        total_walkers,
    )
    plot_charge_surface(
        charge_green,
        grid_size,
        start_row,
        start_column,
        total_walkers,
    )
    plot_boundary_heatmap(
        boundary_green,
        grid_size,
        start_row,
        start_column,
        total_walkers,
    )

    print(
        "Saved "
        f"green_charge_heatmap_N{grid_size}_r{start_row}_c{start_column}_w{total_walkers}x.png"
    )
    print(
        "Saved "
        f"green_charge_surface_N{grid_size}_r{start_row}_c{start_column}_w{total_walkers}x.png"
    )
    print(
        "Saved "
        f"green_boundary_heatmap_N{grid_size}_r{start_row}_c{start_column}_w{total_walkers}x.png"
    )


if __name__ == "__main__":
    main()
