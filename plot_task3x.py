"""Plot saved Green's function and error data for Task 3 Cython runs."""

#pylint score:8.95

import argparse

import matplotlib.pyplot as plt
import numpy as np
from openpyxl import Workbook


def get_task3_sites(grid_size):
    """Return the three Task 3 start sites in grid indices."""

    return {
        "centre": (grid_size // 2, grid_size // 2),
        "corner": (2, 2),
        "face": (grid_size // 2, 2),
    }


def build_file_stub(grid_size, start_row, start_column, total_walkers):
    """Match the existing Cython filename convention."""

    return f"N{grid_size}_r{start_row}_c{start_column}_w{total_walkers}"


def get_case_position(start_row, start_column, grid_spacing):
    """Return physical source coordinates in metres and centimetres."""

    x_m = start_column * grid_spacing
    y_m = start_row * grid_spacing

    return {
        "x_m": x_m,
        "y_m": y_m,
        "x_cm": 100.0 * x_m,
        "y_cm": 100.0 * y_m,
    }


def load_case(case_name, grid_size, grid_spacing, total_walkers):
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

    position = get_case_position(start_row, start_column, grid_spacing)

    return {
        "case_name": case_name,
        "start_row": start_row,
        "start_column": start_column,
        "x_m": position["x_m"],
        "y_m": position["y_m"],
        "x_cm": position["x_cm"],
        "y_cm": position["y_cm"],
        "file_stub": file_stub,
        "charge_green": charge_green,
        "charge_std": charge_std,
        "boundary_green": boundary_green,
        "boundary_std": boundary_std,
    }


def make_extent(grid_size, grid_spacing):
    """Return the plot extent in metres for a square of side length 1 m."""

    side_length = (grid_size - 1) * grid_spacing
    return [0.0, side_length, 0.0, side_length]


def save_2d_field(array, title, colorbar_label, filename, cmap, extent):
    """Save a single 2D field plot using physical axes in metres."""

    plt.figure(figsize=(6, 5))
    image = plt.imshow(array, origin="lower", cmap=cmap, extent=extent, aspect="equal")
    plt.colorbar(image, label=colorbar_label)
    plt.title(title)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def save_3d_field(array, title, zlabel, filename, cmap, grid_size, grid_spacing):
    """Save a single 3D surface plot using physical axes in metres."""

    x = np.arange(grid_size, dtype=np.float64) * grid_spacing
    y = np.arange(grid_size, dtype=np.float64) * grid_spacing
    x_grid, y_grid = np.meshgrid(x, y, indexing="xy")

    figure = plt.figure(figsize=(7, 5.5))
    axis = figure.add_subplot(111, projection="3d")
    surface = axis.plot_surface(x_grid, y_grid, array, cmap=cmap, linewidth=0.0)

    axis.set_title(title)
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_zlabel(zlabel)
    figure.colorbar(surface, ax=axis, shrink=0.7, pad=0.12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_charge_heatmap(case_data, grid_size, grid_spacing, total_walkers):
    """Save the 2D charge Green's function heatmap."""

    save_2d_field(
        case_data["charge_green"],
        (
            "Charge Green's function "
            f"({case_data['case_name']}, "
            f"source=({case_data['x_cm']:.1f} cm, {case_data['y_cm']:.1f} cm), "
            f"walkers={total_walkers}, cython)"
        ),
        "Charge Green's function",
        f"green_charge_heatmap_{case_data['file_stub']}x.png",
        "viridis",
        make_extent(grid_size, grid_spacing),
    )


def plot_charge_std_heatmap(case_data, grid_size, grid_spacing, total_walkers):
    """Save the 2D charge standard deviation heatmap."""

    save_2d_field(
        case_data["charge_std"],
        (
            "Charge Green's function standard deviation "
            f"({case_data['case_name']}, "
            f"source=({case_data['x_cm']:.1f} cm, {case_data['y_cm']:.1f} cm), "
            f"walkers={total_walkers}, cython)"
        ),
        "Charge Green's function standard deviation",
        f"green_charge_std_heatmap_{case_data['file_stub']}x.png",
        "viridis",
        make_extent(grid_size, grid_spacing),
    )


def plot_boundary_heatmap(case_data, grid_size, grid_spacing, total_walkers):
    """Save the 2D boundary Green's function heatmap."""

    save_2d_field(
        case_data["boundary_green"],
        (
            "Boundary Green's function "
            f"({case_data['case_name']}, "
            f"source=({case_data['x_cm']:.1f} cm, {case_data['y_cm']:.1f} cm), "
            f"walkers={total_walkers}, cython)"
        ),
        "Boundary Green's function",
        f"green_boundary_heatmap_{case_data['file_stub']}x.png",
        "magma",
        make_extent(grid_size, grid_spacing),
    )


def plot_boundary_std_heatmap(case_data, grid_size, grid_spacing, total_walkers):
    """Save the 2D boundary standard deviation heatmap."""

    save_2d_field(
        case_data["boundary_std"],
        (
            "Boundary Green's function standard deviation "
            f"({case_data['case_name']}, "
            f"source=({case_data['x_cm']:.1f} cm, {case_data['y_cm']:.1f} cm), "
            f"walkers={total_walkers}, cython)"
        ),
        "Boundary Green's function standard deviation",
        f"green_boundary_std_heatmap_{case_data['file_stub']}x.png",
        "magma",
        make_extent(grid_size, grid_spacing),
    )


def plot_charge_surface(case_data, grid_size, grid_spacing, total_walkers):
    """Save the 3D charge Green's function surface."""

    save_3d_field(
        case_data["charge_green"],
        (
            "Charge Green's function surface "
            f"({case_data['case_name']}, "
            f"source=({case_data['x_cm']:.1f} cm, {case_data['y_cm']:.1f} cm), "
            f"walkers={total_walkers}, cython)"
        ),
        "Charge Green's function",
        f"green_charge_surface_{case_data['file_stub']}x.png",
        "viridis",
        grid_size,
        grid_spacing,
    )


def plot_charge_std_surface(case_data, grid_size, grid_spacing, total_walkers):
    """Save the 3D charge standard deviation surface."""

    save_3d_field(
        case_data["charge_std"],
        (
            "Charge Green's function standard deviation surface "
            f"({case_data['case_name']}, "
            f"source=({case_data['x_cm']:.1f} cm, {case_data['y_cm']:.1f} cm), "
            f"walkers={total_walkers}, cython)"
        ),
        "Charge standard deviation",
        f"green_charge_std_surface_{case_data['file_stub']}x.png",
        "viridis",
        grid_size,
        grid_spacing,
    )


def plot_boundary_surface(case_data, grid_size, grid_spacing, total_walkers):
    """Save the 3D boundary Green's function surface."""

    save_3d_field(
        case_data["boundary_green"],
        (
            "Boundary Green's function surface "
            f"({case_data['case_name']}, "
            f"source=({case_data['x_cm']:.1f} cm, {case_data['y_cm']:.1f} cm), "
            f"walkers={total_walkers}, cython)"
        ),
        "Boundary Green's function",
        f"green_boundary_surface_{case_data['file_stub']}x.png",
        "magma",
        grid_size,
        grid_spacing,
    )


def plot_boundary_std_surface(case_data, grid_size, grid_spacing, total_walkers):
    """Save the 3D boundary standard deviation surface."""

    save_3d_field(
        case_data["boundary_std"],
        (
            "Boundary Green's function standard deviation surface "
            f"({case_data['case_name']}, "
            f"source=({case_data['x_cm']:.1f} cm, {case_data['y_cm']:.1f} cm), "
            f"walkers={total_walkers}, cython)"
        ),
        "Boundary standard deviation",
        f"green_boundary_std_surface_{case_data['file_stub']}x.png",
        "magma",
        grid_size,
        grid_spacing,
    )


def plot_combined_2x2(case_data, grid_size, grid_spacing, total_walkers):
    """Save one compact figure showing value and error together."""

    extent = make_extent(grid_size, grid_spacing)

    figure, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)

    image00 = axes[0, 0].imshow(
        case_data["charge_green"],
        origin="lower",
        cmap="viridis",
        extent=extent,
        aspect="equal",
    )
    axes[0, 0].set_title("Charge Green's function")
    axes[0, 0].set_xlabel("x (m)")
    axes[0, 0].set_ylabel("y (m)")
    figure.colorbar(image00, ax=axes[0, 0], shrink=0.85)

    image01 = axes[0, 1].imshow(
        case_data["charge_std"],
        origin="lower",
        cmap="viridis",
        extent=extent,
        aspect="equal",
    )
    axes[0, 1].set_title("Charge standard deviation")
    axes[0, 1].set_xlabel("x (m)")
    axes[0, 1].set_ylabel("y (m)")
    figure.colorbar(image01, ax=axes[0, 1], shrink=0.85)

    image10 = axes[1, 0].imshow(
        case_data["boundary_green"],
        origin="lower",
        cmap="magma",
        extent=extent,
        aspect="equal",
    )
    axes[1, 0].set_title("Boundary Green's function")
    axes[1, 0].set_xlabel("x (m)")
    axes[1, 0].set_ylabel("y (m)")
    figure.colorbar(image10, ax=axes[1, 0], shrink=0.85)

    image11 = axes[1, 1].imshow(
        case_data["boundary_std"],
        origin="lower",
        cmap="magma",
        extent=extent,
        aspect="equal",
    )
    axes[1, 1].set_title("Boundary standard deviation")
    axes[1, 1].set_xlabel("x (m)")
    axes[1, 1].set_ylabel("y (m)")
    figure.colorbar(image11, ax=axes[1, 1], shrink=0.85)

    figure.suptitle(
        "Task 3 fields and errors "
        f"({case_data['case_name']}, "
        f"source=({case_data['x_cm']:.1f} cm, {case_data['y_cm']:.1f} cm), "
        f"walkers={total_walkers}, cython)"
    )

    figure.savefig(f"task3_combined_{case_data['file_stub']}x.png", dpi=300)
    plt.close(figure)


def plot_comparison(
    case_data_list,
    key,
    cmap,
    colorbar_label,
    title,
    filename,
    grid_size,
    grid_spacing,
):
    """Save a three-panel comparison across the Task 3 start sites."""

    extent = make_extent(grid_size, grid_spacing)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    image = None

    for axis, case_data in zip(axes, case_data_list):
        image = axis.imshow(
            case_data[key],
            origin="lower",
            cmap=cmap,
            extent=extent,
            aspect="equal",
        )
        axis.set_title(
            f"{case_data['case_name']}\n"
            f"source=({case_data['x_cm']:.1f} cm, {case_data['y_cm']:.1f} cm)"
        )
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")

    figure.colorbar(image, ax=axes, shrink=0.9, label=colorbar_label)
    figure.suptitle(title)
    figure.savefig(filename, dpi=300)
    plt.close(figure)
    print(f"Saved {filename}")


def build_error_summary(case_data, total_walkers):
    """Return scalar error summary values for one case."""

    sqrt_walkers = np.sqrt(float(total_walkers))

    charge_se = case_data["charge_std"] / sqrt_walkers
    boundary_se = case_data["boundary_std"] / sqrt_walkers

    start_row = case_data["start_row"]
    start_column = case_data["start_column"]

    return {
        "case_name": case_data["case_name"],
        "start_row": start_row,
        "start_column": start_column,
        "x_m": case_data["x_m"],
        "y_m": case_data["y_m"],
        "x_cm": case_data["x_cm"],
        "y_cm": case_data["y_cm"],
        "charge_std_mean": float(np.mean(case_data["charge_std"])),
        "charge_std_max": float(np.max(case_data["charge_std"])),
        "charge_std_at_start": float(case_data["charge_std"][start_row, start_column]),
        "charge_se_mean": float(np.mean(charge_se)),
        "charge_se_max": float(np.max(charge_se)),
        "charge_se_at_start": float(charge_se[start_row, start_column]),
        "boundary_std_mean": float(np.mean(case_data["boundary_std"])),
        "boundary_std_max": float(np.max(case_data["boundary_std"])),
        "boundary_std_at_start": float(case_data["boundary_std"][start_row, start_column]),
        "boundary_se_mean": float(np.mean(boundary_se)),
        "boundary_se_max": float(np.max(boundary_se)),
        "boundary_se_at_start": float(boundary_se[start_row, start_column]),
    }


def print_error_summary(summary):
    """Print one compact Task 3 error summary."""

    print()
    print("=" * 80)
    print(
        f"Case: {summary['case_name']} "
        f"({summary['x_cm']:.1f} cm, {summary['y_cm']:.1f} cm)"
    )
    print("=" * 80)
    print(f"Charge std mean:        {summary['charge_std_mean']:.8e}")
    print(f"Charge std max:         {summary['charge_std_max']:.8e}")
    print(f"Charge std at source:   {summary['charge_std_at_start']:.8e}")
    print(f"Charge se mean:         {summary['charge_se_mean']:.8e}")
    print(f"Charge se max:          {summary['charge_se_max']:.8e}")
    print(f"Charge se at source:    {summary['charge_se_at_start']:.8e}")
    print(f"Boundary std mean:      {summary['boundary_std_mean']:.8e}")
    print(f"Boundary std max:       {summary['boundary_std_max']:.8e}")
    print(f"Boundary std at source: {summary['boundary_std_at_start']:.8e}")
    print(f"Boundary se mean:       {summary['boundary_se_mean']:.8e}")
    print(f"Boundary se max:        {summary['boundary_se_max']:.8e}")
    print(f"Boundary se at source:  {summary['boundary_se_at_start']:.8e}")


def save_error_spreadsheet(summary_list, grid_size, grid_spacing, total_walkers):
    """Save Task 3 scalar error summaries to an Excel workbook."""

    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "task3_error_summary"

    worksheet.append(
        [
            "case",
            "grid_size",
            "grid_spacing_m",
            "side_length_m",
            "walkers",
            "start_row",
            "start_column",
            "x_m",
            "y_m",
            "x_cm",
            "y_cm",
            "charge_std_mean",
            "charge_std_max",
            "charge_std_at_start",
            "charge_se_mean",
            "charge_se_max",
            "charge_se_at_start",
            "boundary_std_mean",
            "boundary_std_max",
            "boundary_std_at_start",
            "boundary_se_mean",
            "boundary_se_max",
            "boundary_se_at_start",
        ]
    )

    side_length = (grid_size - 1) * grid_spacing

    for summary in summary_list:
        worksheet.append(
            [
                summary["case_name"],
                grid_size,
                grid_spacing,
                side_length,
                total_walkers,
                summary["start_row"],
                summary["start_column"],
                summary["x_m"],
                summary["y_m"],
                summary["x_cm"],
                summary["y_cm"],
                summary["charge_std_mean"],
                summary["charge_std_max"],
                summary["charge_std_at_start"],
                summary["charge_se_mean"],
                summary["charge_se_max"],
                summary["charge_se_at_start"],
                summary["boundary_std_mean"],
                summary["boundary_std_max"],
                summary["boundary_std_at_start"],
                summary["boundary_se_mean"],
                summary["boundary_se_max"],
                summary["boundary_se_at_start"],
            ]
        )

    filename = f"task3_error_summary_N{grid_size}_w{total_walkers}x.xlsx"
    workbook.save(filename)
    print(f"Saved {filename}")


def plot_case(case_data, grid_size, grid_spacing, total_walkers):
    """Generate all single-case Task 3 plots and return its summary."""

    plot_charge_heatmap(case_data, grid_size, grid_spacing, total_walkers)
    plot_charge_std_heatmap(case_data, grid_size, grid_spacing, total_walkers)
    plot_boundary_heatmap(case_data, grid_size, grid_spacing, total_walkers)
    plot_boundary_std_heatmap(case_data, grid_size, grid_spacing, total_walkers)

    plot_charge_surface(case_data, grid_size, grid_spacing, total_walkers)
    plot_charge_std_surface(case_data, grid_size, grid_spacing, total_walkers)
    plot_boundary_surface(case_data, grid_size, grid_spacing, total_walkers)
    plot_boundary_std_surface(case_data, grid_size, grid_spacing, total_walkers)

    plot_combined_2x2(case_data, grid_size, grid_spacing, total_walkers)

    print(f"Saved green_charge_heatmap_{case_data['file_stub']}x.png")
    print(f"Saved green_charge_std_heatmap_{case_data['file_stub']}x.png")
    print(f"Saved green_boundary_heatmap_{case_data['file_stub']}x.png")
    print(f"Saved green_boundary_std_heatmap_{case_data['file_stub']}x.png")
    print(f"Saved green_charge_surface_{case_data['file_stub']}x.png")
    print(f"Saved green_charge_std_surface_{case_data['file_stub']}x.png")
    print(f"Saved green_boundary_surface_{case_data['file_stub']}x.png")
    print(f"Saved green_boundary_std_surface_{case_data['file_stub']}x.png")
    print(f"Saved task3_combined_{case_data['file_stub']}x.png")

    summary = build_error_summary(case_data, total_walkers)
    print_error_summary(summary)

    return summary


def main():
    """Load saved Task 3 Cython outputs and generate figures and summaries."""

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
        "--grid-spacing",
        type=float,
        default=0.01,
        help="Grid spacing in metres.",
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
    summary_list = []

    for case_name in case_names:
        case_data = load_case(case_name, args.grid_size, args.grid_spacing, args.walkers)
        summary = plot_case(case_data, args.grid_size, args.grid_spacing, args.walkers)
        case_data_list.append(case_data)
        summary_list.append(summary)

    save_error_spreadsheet(summary_list, args.grid_size, args.grid_spacing, args.walkers)

    if len(case_data_list) == 3:
        plot_comparison(
            case_data_list,
            "charge_green",
            "viridis",
            "Charge Green's function",
            f"Task 3 charge comparison (N={args.grid_size}, walkers={args.walkers}, cython)",
            f"task3_charge_comparison_N{args.grid_size}_w{args.walkers}x.png",
            args.grid_size,
            args.grid_spacing,
        )
        plot_comparison(
            case_data_list,
            "charge_std",
            "viridis",
            "Charge Green's function standard deviation",
            f"Task 3 charge standard deviation comparison (N={args.grid_size}, walkers={args.walkers}, cython)",
            f"task3_charge_std_comparison_N{args.grid_size}_w{args.walkers}x.png",
            args.grid_size,
            args.grid_spacing,
        )
        plot_comparison(
            case_data_list,
            "boundary_green",
            "magma",
            "Boundary Green's function",
            f"Task 3 boundary comparison (N={args.grid_size}, walkers={args.walkers}, cython)",
            f"task3_boundary_comparison_N{args.grid_size}_w{args.walkers}x.png",
            args.grid_size,
            args.grid_spacing,
        )
        plot_comparison(
            case_data_list,
            "boundary_std",
            "magma",
            "Boundary Green's function standard deviation",
            f"Task 3 boundary standard deviation comparison (N={args.grid_size}, walkers={args.walkers}, cython)",
            f"task3_boundary_std_comparison_N{args.grid_size}_w{args.walkers}x.png",
            args.grid_size,
            args.grid_spacing,
        )


if __name__ == "__main__":
    main()
