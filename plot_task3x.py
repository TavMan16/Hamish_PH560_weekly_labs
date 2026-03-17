"""Plot saved Green's function and error data for Task 3 Cython runs."""

import argparse

import matplotlib.pyplot as plt
import numpy as np


def get_task3_sites(grid_size):
    """Return the three Task 3 start sites."""

    return {
        "centre": (grid_size // 2, grid_size // 2),
        "corner": (2, 2),
        "face": (2, grid_size // 2),
    }


def build_file_stub(grid_size, start_row, start_column, total_walkers):
    """Match the existing Cython filename convention."""

    return f"N{grid_size}_r{start_row}_c{start_column}_w{total_walkers}"


def load_case(case_name, grid_size, total_walkers):
    """Load one saved Task 3 case from disk."""

    start_row, start_column = get_task3_sites(grid_size)[case_name]
    file_stub = build_file_stub(grid_size, start_row, start_column, total_walkers)

    charge_green = np.load(f"green_charge_{file_stub}x.npy")
    charge_std = np.load(f"green_charge_std_{file_stub}x.npy")
    boundary_green = np.load(f"green_boundary_{file_stub}x.npy")
    boundary_std = np.load(f"green_boundary_std_{file_stub}x.npy")

    print(f"Loaded green_charge_{file_stub}x.npy")
    print(f"Loaded green_charge_std_{file_stub}x.npy")
    print(f"Loaded green_boundary_{file_stub}x.npy")
    print(f"Loaded green_boundary_std_{file_stub}x.npy")

    return {
        "case_name": case_name,
        "start_row": start_row,
        "start_column": start_column,
        "file_stub": file_stub,
        "charge_green": charge_green,
        "charge_std": charge_std,
        "boundary_green": boundary_green,
        "boundary_std": boundary_std,
    }


def save_2d_field(array, title, colorbar_label, filename, cmap):
    """Save a single 2D field plot."""

    plt.figure(figsize=(6, 5))
    image = plt.imshow(array, origin="lower", cmap=cmap)
    plt.colorbar(image, label=colorbar_label)
    plt.title(title)
    plt.xlabel("Column index")
    plt.ylabel("Row index")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def save_3d_field(array, title, zlabel, filename, cmap):
    """Save a single 3D surface plot."""

    grid_size = array.shape[0]
    rows, cols = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")

    figure = plt.figure(figsize=(7, 5.5))
    axis = figure.add_subplot(111, projection="3d")

    surface = axis.plot_surface(
        cols,
        rows,
        array,
        cmap=cmap,
        linewidth=0,
        antialiased=False,
    )

    figure.colorbar(surface, ax=axis, shrink=0.75, pad=0.1)
    axis.set_title(title)
    axis.set_xlabel("Column index")
    axis.set_ylabel("Row index")
    axis.set_zlabel(zlabel)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(figure)


def plot_charge_heatmap(case_data, grid_size, total_walkers):
    """Save the 2D charge Green's function."""

    save_2d_field(
        case_data["charge_green"],
        (
            "Charge Green's function "
            f"({case_data['case_name']}, N={grid_size}, "
            f"start=({case_data['start_row']}, {case_data['start_column']}), "
            f"walkers={total_walkers}, cython)"
        ),
        "Charge Green's function",
        f"green_charge_heatmap_{case_data['file_stub']}x.png",
        "viridis",
    )


def plot_charge_std_heatmap(case_data, grid_size, total_walkers):
    """Save the 2D charge error field."""

    save_2d_field(
        case_data["charge_std"],
        (
            "Charge Green's function error "
            f"({case_data['case_name']}, N={grid_size}, "
            f"start=({case_data['start_row']}, {case_data['start_column']}), "
            f"walkers={total_walkers}, cython)"
        ),
        "Charge Green's function standard deviation",
        f"green_charge_std_heatmap_{case_data['file_stub']}x.png",
        "viridis",
    )


def plot_boundary_heatmap(case_data, grid_size, total_walkers):
    """Save the 2D boundary Green's function."""

    save_2d_field(
        case_data["boundary_green"],
        (
            "Boundary Green's function "
            f"({case_data['case_name']}, N={grid_size}, "
            f"start=({case_data['start_row']}, {case_data['start_column']}), "
            f"walkers={total_walkers}, cython)"
        ),
        "Boundary Green's function",
        f"green_boundary_heatmap_{case_data['file_stub']}x.png",
        "magma",
    )


def plot_boundary_std_heatmap(case_data, grid_size, total_walkers):
    """Save the 2D boundary error field."""

    save_2d_field(
        case_data["boundary_std"],
        (
            "Boundary Green's function error "
            f"({case_data['case_name']}, N={grid_size}, "
            f"start=({case_data['start_row']}, {case_data['start_column']}), "
            f"walkers={total_walkers}, cython)"
        ),
        "Boundary Green's function standard deviation",
        f"green_boundary_std_heatmap_{case_data['file_stub']}x.png",
        "magma",
    )


def plot_charge_surface(case_data, grid_size, total_walkers):
    """Save the 3D charge Green's function."""

    save_3d_field(
        case_data["charge_green"],
        (
            "Charge Green's function surface "
            f"({case_data['case_name']}, N={grid_size}, "
            f"start=({case_data['start_row']}, {case_data['start_column']}), "
            f"walkers={total_walkers}, cython)"
        ),
        "Charge Green's function",
        f"green_charge_surface_{case_data['file_stub']}x.png",
        "viridis",
    )


def plot_charge_std_surface(case_data, grid_size, total_walkers):
    """Save the 3D charge error surface."""

    save_3d_field(
        case_data["charge_std"],
        (
            "Charge Green's function error surface "
            f"({case_data['case_name']}, N={grid_size}, "
            f"start=({case_data['start_row']}, {case_data['start_column']}), "
            f"walkers={total_walkers}, cython)"
        ),
        "Charge standard deviation",
        f"green_charge_std_surface_{case_data['file_stub']}x.png",
        "viridis",
    )


def plot_boundary_surface(case_data, grid_size, total_walkers):
    """Save the 3D boundary Green's function."""

    save_3d_field(
        case_data["boundary_green"],
        (
            "Boundary Green's function surface "
            f"({case_data['case_name']}, N={grid_size}, "
            f"start=({case_data['start_row']}, {case_data['start_column']}), "
            f"walkers={total_walkers}, cython)"
        ),
        "Boundary Green's function",
        f"green_boundary_surface_{case_data['file_stub']}x.png",
        "magma",
    )


def plot_boundary_std_surface(case_data, grid_size, total_walkers):
    """Save the 3D boundary error surface."""

    save_3d_field(
        case_data["boundary_std"],
        (
            "Boundary Green's function error surface "
            f"({case_data['case_name']}, N={grid_size}, "
            f"start=({case_data['start_row']}, {case_data['start_column']}), "
            f"walkers={total_walkers}, cython)"
        ),
        "Boundary standard deviation",
        f"green_boundary_std_surface_{case_data['file_stub']}x.png",
        "magma",
    )


def plot_combined_2x2(case_data, grid_size, total_walkers):
    """Save one compact figure showing value and error together."""

    figure, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)

    image00 = axes[0, 0].imshow(case_data["charge_green"], origin="lower", cmap="viridis")
    axes[0, 0].set_title("Charge Green's function")
    axes[0, 0].set_xlabel("Column index")
    axes[0, 0].set_ylabel("Row index")
    figure.colorbar(image00, ax=axes[0, 0], shrink=0.85)

    image01 = axes[0, 1].imshow(case_data["charge_std"], origin="lower", cmap="viridis")
    axes[0, 1].set_title("Charge error")
    axes[0, 1].set_xlabel("Column index")
    axes[0, 1].set_ylabel("Row index")
    figure.colorbar(image01, ax=axes[0, 1], shrink=0.85)

    image10 = axes[1, 0].imshow(case_data["boundary_green"], origin="lower", cmap="magma")
    axes[1, 0].set_title("Boundary Green's function")
    axes[1, 0].set_xlabel("Column index")
    axes[1, 0].set_ylabel("Row index")
    figure.colorbar(image10, ax=axes[1, 0], shrink=0.85)

    image11 = axes[1, 1].imshow(case_data["boundary_std"], origin="lower", cmap="magma")
    axes[1, 1].set_title("Boundary error")
    axes[1, 1].set_xlabel("Column index")
    axes[1, 1].set_ylabel("Row index")
    figure.colorbar(image11, ax=axes[1, 1], shrink=0.85)

    figure.suptitle(
        f"Task 3 fields and errors ({case_data['case_name']}, "
        f"N={grid_size}, start=({case_data['start_row']}, {case_data['start_column']}), "
        f"walkers={total_walkers}, cython)"
    )

    figure.savefig(f"task3_combined_{case_data['file_stub']}x.png", dpi=300)
    plt.close(figure)


def plot_comparison(case_data_list, key, cmap, colorbar_label, title, filename):
    """Save a three-panel comparison across the Task 3 start sites."""

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    image = None

    for axis, case_data in zip(axes, case_data_list):
        image = axis.imshow(case_data[key], origin="lower", cmap=cmap)
        axis.set_title(
            f"{case_data['case_name']}\n"
            f"start=({case_data['start_row']}, {case_data['start_column']})"
        )
        axis.set_xlabel("Column index")
        axis.set_ylabel("Row index")

    figure.colorbar(image, ax=axes, shrink=0.9, label=colorbar_label)
    figure.suptitle(title)
    figure.savefig(filename, dpi=300)
    plt.close(figure)


def print_error_summary(case_data):
    """Print simple scalar summaries for the report."""

    charge_std = case_data["charge_std"]
    boundary_std = case_data["boundary_std"]
    start_row = case_data["start_row"]
    start_column = case_data["start_column"]

    print()
    print(f"Error summary for {case_data['case_name']}:")
    print(f"Charge std mean: {np.mean(charge_std):.6e}")
    print(f"Charge std max: {np.max(charge_std):.6e}")
    print(f"Charge std at start: {charge_std[start_row, start_column]:.6e}")
    print(f"Boundary std mean: {np.mean(boundary_std):.6e}")
    print(f"Boundary std max: {np.max(boundary_std):.6e}")
    print(f"Boundary std at start: {boundary_std[start_row, start_column]:.6e}")


def plot_case(case_data, grid_size, total_walkers):
    """Generate all single-case Task 3 plots."""

    plot_charge_heatmap(case_data, grid_size, total_walkers)
    plot_charge_std_heatmap(case_data, grid_size, total_walkers)
    plot_boundary_heatmap(case_data, grid_size, total_walkers)
    plot_boundary_std_heatmap(case_data, grid_size, total_walkers)

    plot_charge_surface(case_data, grid_size, total_walkers)
    plot_charge_std_surface(case_data, grid_size, total_walkers)
    plot_boundary_surface(case_data, grid_size, total_walkers)
    plot_boundary_std_surface(case_data, grid_size, total_walkers)

    plot_combined_2x2(case_data, grid_size, total_walkers)

    print(f"Saved green_charge_heatmap_{case_data['file_stub']}x.png")
    print(f"Saved green_charge_std_heatmap_{case_data['file_stub']}x.png")
    print(f"Saved green_boundary_heatmap_{case_data['file_stub']}x.png")
    print(f"Saved green_boundary_std_heatmap_{case_data['file_stub']}x.png")
    print(f"Saved green_charge_surface_{case_data['file_stub']}x.png")
    print(f"Saved green_charge_std_surface_{case_data['file_stub']}x.png")
    print(f"Saved green_boundary_surface_{case_data['file_stub']}x.png")
    print(f"Saved green_boundary_std_surface_{case_data['file_stub']}x.png")
    print(f"Saved task3_combined_{case_data['file_stub']}x.png")

    print_error_summary(case_data)


def main():
    """Load saved Task 3 Cython outputs and generate figures."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=["centre", "corner", "face", "all"],
        default="all",
        help="Task 3 start site to plot.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=101,
        help="Number of grid points in each direction.",
    )
    parser.add_argument(
        "--walkers",
        type=int,
        default=1000000,
        help="Total number of walkers used in the saved files.",
    )
    args = parser.parse_args()

    if args.case == "all":
        case_names = ["centre", "corner", "face"]
    else:
        case_names = [args.case]

    case_data_list = []

    for case_name in case_names:
        case_data = load_case(case_name, args.grid_size, args.walkers)
        plot_case(case_data, args.grid_size, args.walkers)
        case_data_list.append(case_data)

    if len(case_data_list) == 3:
        plot_comparison(
            case_data_list,
            "charge_green",
            "viridis",
            "Charge Green's function",
            f"Task 3 charge comparison (N={args.grid_size}, walkers={args.walkers}, cython)",
            f"task3_charge_comparison_N{args.grid_size}_w{args.walkers}x.png",
        )
        plot_comparison(
            case_data_list,
            "charge_std",
            "viridis",
            "Charge Green's function standard deviation",
            f"Task 3 charge error comparison (N={args.grid_size}, walkers={args.walkers}, cython)",
            f"task3_charge_std_comparison_N{args.grid_size}_w{args.walkers}x.png",
        )
        plot_comparison(
            case_data_list,
            "boundary_green",
            "magma",
            "Boundary Green's function",
            f"Task 3 boundary comparison (N={args.grid_size}, walkers={args.walkers}, cython)",
            f"task3_boundary_comparison_N{args.grid_size}_w{args.walkers}x.png",
        )
        plot_comparison(
            case_data_list,
            "boundary_std",
            "magma",
            "Boundary Green's function standard deviation",
            f"Task 3 boundary error comparison (N={args.grid_size}, walkers={args.walkers}, cython)",
            f"task3_boundary_std_comparison_N{args.grid_size}_w{args.walkers}x.png",
        )

        print(f"Saved task3_charge_comparison_N{args.grid_size}_w{args.walkers}x.png")
        print(f"Saved task3_charge_std_comparison_N{args.grid_size}_w{args.walkers}x.png")
        print(f"Saved task3_boundary_comparison_N{args.grid_size}_w{args.walkers}x.png")
        print(f"Saved task3_boundary_std_comparison_N{args.grid_size}_w{args.walkers}x.png")


if __name__ == "__main__":
    main()
