"""Reconstruct Task 4 potentials from saved Green's function arrays."""

import argparse
import csv
import math

import numpy as np


def get_task3_sites(grid_size):
    """Return the three Task 3 evaluation sites."""

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


def load_green_case(case_name, grid_size, grid_spacing, total_walkers):
    """Load one saved Green's function case from disk."""

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


def build_boundary_values(grid_size, boundary_name):
    """Return the boundary-potential field for one Task 4 boundary condition."""

    boundary_values = np.zeros((grid_size, grid_size), dtype=np.float64)

    if boundary_name == "uniform_100":
        top_value = 100.0
        bottom_value = 100.0
        left_value = 100.0
        right_value = 100.0
    elif boundary_name == "tb_100_lr_minus100":
        top_value = 100.0
        bottom_value = 100.0
        left_value = -100.0
        right_value = -100.0
    elif boundary_name == "tl_200_b_0_r_minus400":
        top_value = 200.0
        bottom_value = 0.0
        left_value = 200.0
        right_value = -400.0
    else:
        raise ValueError(f"Unknown boundary condition: {boundary_name}")

    boundary_values[0, :] = bottom_value
    boundary_values[-1, :] = top_value
    boundary_values[:, 0] = left_value
    boundary_values[:, -1] = right_value

    return boundary_values


def build_charge_density(grid_size, grid_spacing, charge_name):
    """Return the charge-density field for one Task 4 charge distribution."""

    if charge_name == "zero":
        return np.zeros((grid_size, grid_size), dtype=np.float64)

    if charge_name == "uniform_10":
        return np.full((grid_size, grid_size), 10.0, dtype=np.float64)

    y = np.arange(grid_size, dtype=np.float64) * grid_spacing
    x = np.arange(grid_size, dtype=np.float64) * grid_spacing
    y_grid, x_grid = np.meshgrid(y, x, indexing="ij")

    if charge_name == "top_to_bottom_gradient":
        side_length = (grid_size - 1) * grid_spacing
        return y_grid / side_length

    if charge_name == "exp_centre":
        centre = 0.5 * (grid_size - 1) * grid_spacing
        radius = np.sqrt((x_grid - centre) ** 2 + (y_grid - centre) ** 2)
        return np.exp(-10.0 * radius)

    raise ValueError(f"Unknown charge distribution: {charge_name}")


def boundary_mask(grid_size):
    """Return a boolean mask selecting the outer boundary sites."""

    mask = np.zeros((grid_size, grid_size), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def reconstruct_potential(case_data, boundary_values, charge_density, total_walkers):
    """
    Reconstruct the potential and propagated standard error for one case.

    The saved std arrays are sample standard deviations across walkers, so they are
    converted to standard errors of the Green's function means before propagation.
    """

    mask = boundary_mask(boundary_values.shape[0])

    boundary_weights = boundary_values[mask]
    boundary_green = case_data["boundary_green"][mask]
    boundary_se = case_data["boundary_std"][mask] / math.sqrt(total_walkers)

    charge_green = case_data["charge_green"]
    charge_se = case_data["charge_std"] / math.sqrt(total_walkers)

    boundary_contribution = np.sum(boundary_green * boundary_weights)
    charge_contribution = np.sum(charge_green * charge_density)

    boundary_error = np.sqrt(np.sum((boundary_se * boundary_weights) ** 2))
    charge_error = np.sqrt(np.sum((charge_se * charge_density) ** 2))

    total_potential = boundary_contribution + charge_contribution
    total_error = np.sqrt(boundary_error ** 2 + charge_error ** 2)

    return {
        "boundary_contribution": float(boundary_contribution),
        "charge_contribution": float(charge_contribution),
        "total_potential": float(total_potential),
        "boundary_error": float(boundary_error),
        "charge_error": float(charge_error),
        "total_error": float(total_error),
    }


def boundary_label(boundary_name):
    """Return a printable label for a boundary condition."""

    labels = {
        "uniform_100": "all edges +100 V",
        "tb_100_lr_minus100": "top/bottom +100 V, left/right -100 V",
        "tl_200_b_0_r_minus400": "top/left +200 V, bottom 0 V, right -400 V",
    }
    return labels[boundary_name]


def charge_label(charge_name):
    """Return a printable label for a charge distribution."""

    labels = {
        "zero": "zero charge",
        "uniform_10": "uniform 10 C m^-2",
        "top_to_bottom_gradient": "top-to-bottom gradient",
        "exp_centre": "exp(-10r) at centre",
    }
    return labels[charge_name]


def print_result(case_data, boundary_name, charge_name, result):
    """Print one reconstructed Task 4 result block."""

    print()
    print(f"Boundary: {boundary_label(boundary_name)}")
    print(f"Charge:   {charge_label(charge_name)}")
    print(f"Boundary contribution: {result['boundary_contribution']:.8e}")
    print(f"Charge contribution:   {result['charge_contribution']:.8e}")
    print(f"Total potential:       {result['total_potential']:.8e}")
    print(f"Boundary std. error:   {result['boundary_error']:.8e}")
    print(f"Charge std. error:     {result['charge_error']:.8e}")
    print(f"Total std. error:      {result['total_error']:.8e}")


def save_summary_csv(rows, grid_size, total_walkers):
    """Save all Task 4 reconstructed results to one CSV file."""

    filename = f"task4_summary_N{grid_size}_w{total_walkers}x.csv"

    with open(filename, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case",
                "start_row",
                "start_column",
                "x_m",
                "y_m",
                "x_cm",
                "y_cm",
                "boundary_name",
                "charge_name",
                "boundary_contribution",
                "charge_contribution",
                "total_potential",
                "boundary_error",
                "charge_error",
                "total_error",
            ],
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print()
    print(f"Saved {filename}")


def main():
    """Load saved Green's functions, reconstruct Task 4 potentials, and save CSV."""

    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--case",
        choices=["centre", "corner", "face", "all"],
        default="all",
        help="Task 3 evaluation site to reconstruct.",
    )
    args = parser.parse_args()

    if args.case == "all":
        case_names = ["centre", "corner", "face"]
    else:
        case_names = [args.case]

    boundary_names = [
        "uniform_100",
        "tb_100_lr_minus100",
        "tl_200_b_0_r_minus400",
    ]

    charge_names = [
        "zero",
        "uniform_10",
        "top_to_bottom_gradient",
        "exp_centre",
    ]

    summary_rows = []

    for case_name in case_names:
        case_data = load_green_case(
            case_name,
            args.grid_size,
            args.grid_spacing,
            args.walkers,
        )

        print()
        print("=" * 80)
        print(
            f"Case: {case_name} "
            f"(row={case_data['start_row']}, col={case_data['start_column']})"
        )
        print(
            f"Position: ({case_data['x_m']:.3f} m, {case_data['y_m']:.3f} m) "
            f"= ({case_data['x_cm']:.1f} cm, {case_data['y_cm']:.1f} cm)"
        )
        print("=" * 80)

        for boundary_name in boundary_names:
            boundary_values = build_boundary_values(args.grid_size, boundary_name)

            for charge_name in charge_names:
                charge_density = build_charge_density(
                    args.grid_size,
                    args.grid_spacing,
                    charge_name,
                )

                result = reconstruct_potential(
                    case_data,
                    boundary_values,
                    charge_density,
                    args.walkers,
                )

                print_result(case_data, boundary_name, charge_name, result)

                summary_rows.append(
                    {
                        "case": case_data["case_name"],
                        "start_row": case_data["start_row"],
                        "start_column": case_data["start_column"],
                        "x_m": case_data["x_m"],
                        "y_m": case_data["y_m"],
                        "x_cm": case_data["x_cm"],
                        "y_cm": case_data["y_cm"],
                        "boundary_name": boundary_name,
                        "charge_name": charge_name,
                        "boundary_contribution": result["boundary_contribution"],
                        "charge_contribution": result["charge_contribution"],
                        "total_potential": result["total_potential"],
                        "boundary_error": result["boundary_error"],
                        "charge_error": result["charge_error"],
                        "total_error": result["total_error"],
                    }
                )

    save_summary_csv(summary_rows, args.grid_size, args.walkers)


if __name__ == "__main__":
    main()
