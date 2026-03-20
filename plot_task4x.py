"""Plot saved Task 4 reconstructed potentials using grouped error-bar figures."""

#pylint score:9.21

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np


def load_task4_summary(filename):
    """Load the Task 4 CSV summary into a list of dictionaries."""

    rows = []

    with open(filename, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            rows.append(
                {
                    "case": row["case"],
                    "start_row": int(row["start_row"]),
                    "start_column": int(row["start_column"]),
                    "x_m": float(row["x_m"]),
                    "y_m": float(row["y_m"]),
                    "x_cm": float(row["x_cm"]),
                    "y_cm": float(row["y_cm"]),
                    "boundary_name": row["boundary_name"],
                    "charge_name": row["charge_name"],
                    "boundary_contribution": float(row["boundary_contribution"]),
                    "charge_contribution": float(row["charge_contribution"]),
                    "total_potential": float(row["total_potential"]),
                    "boundary_error": float(row["boundary_error"]),
                    "charge_error": float(row["charge_error"]),
                    "total_error": float(row["total_error"]),
                }
            )

    return rows


def get_case_order():
    """Return the fixed plotting order for the three Task 3 sites."""

    return ["centre", "corner", "face"]


def get_boundary_order():
    """Return the fixed plotting order for the Task 4 boundary conditions."""

    return [
        "uniform_100",
        "tb_100_lr_minus100",
        "tl_200_b_0_r_minus400",
    ]


def get_charge_order():
    """Return the fixed plotting order for the Task 4 charge distributions."""

    return [
        "zero",
        "uniform_10",
        "top_to_bottom_gradient",
        "exp_centre",
    ]


def boundary_label(boundary_name):
    """Return a readable label for one boundary condition."""

    labels = {
        "uniform_100": "all +100 V",
        "tb_100_lr_minus100": "TB +100 V\nLR -100 V",
        "tl_200_b_0_r_minus400": "T/L +200 V\nB 0 V, R -400 V",
    }

    return labels[boundary_name]


def charge_label(charge_name):
    """Return a readable label for one charge distribution."""

    labels = {
        "zero": "zero",
        "uniform_10": "uniform 10",
        "top_to_bottom_gradient": "top-bottom grad",
        "exp_centre": "exp centre",
    }

    return labels[charge_name]


def get_case_rows(rows, case_name):
    """Return only the rows belonging to one evaluation site."""

    return [row for row in rows if row["case"] == case_name]


def find_record(rows, case_name, boundary_name, charge_name):
    """Return the unique record for one case/boundary/charge combination."""

    for row in rows:
        if (
            row["case"] == case_name
            and row["boundary_name"] == boundary_name
            and row["charge_name"] == charge_name
        ):
            return row

    raise ValueError(
        "Missing Task 4 record for "
        f"case={case_name}, boundary={boundary_name}, charge={charge_name}"
    )


def build_case_matrices(rows, case_name):
    """Build 2D matrices of Task 4 outputs for one evaluation site."""

    boundary_names = get_boundary_order()
    charge_names = get_charge_order()

    shape = (len(charge_names), len(boundary_names))

    boundary_contribution = np.zeros(shape, dtype=np.float64)
    charge_contribution = np.zeros(shape, dtype=np.float64)
    total_potential = np.zeros(shape, dtype=np.float64)
    total_error = np.zeros(shape, dtype=np.float64)
    boundary_error = np.zeros(shape, dtype=np.float64)
    charge_error = np.zeros(shape, dtype=np.float64)

    for charge_index, charge_name in enumerate(charge_names):
        for boundary_index, boundary_name in enumerate(boundary_names):
            record = find_record(rows, case_name, boundary_name, charge_name)

            boundary_contribution[charge_index, boundary_index] = (
                record["boundary_contribution"]
            )
            charge_contribution[charge_index, boundary_index] = (
                record["charge_contribution"]
            )
            total_potential[charge_index, boundary_index] = record["total_potential"]
            total_error[charge_index, boundary_index] = record["total_error"]
            boundary_error[charge_index, boundary_index] = record["boundary_error"]
            charge_error[charge_index, boundary_index] = record["charge_error"]

    return {
        "boundary_contribution": boundary_contribution,
        "charge_contribution": charge_contribution,
        "total_potential": total_potential,
        "total_error": total_error,
        "boundary_error": boundary_error,
        "charge_error": charge_error,
    }


def setup_style():
    """Apply a simple white plotting style."""

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "lines.linewidth": 1.8,
            "axes.grid": False,
        }
    )


def case_title(case_rows):
    """Return a compact title for one evaluation site."""

    case_info = case_rows[0]

    return (
        f"{case_info['case']}, "
        f"point=({case_info['x_cm']:.1f} cm, {case_info['y_cm']:.1f} cm)"
    )


def draw_grouped_errorbars(axis, values, errors, title):
    """Draw one grouped error-bar plot for one quantity."""

    boundary_names = get_boundary_order()
    charge_names = get_charge_order()

    x = np.arange(len(boundary_names), dtype=np.float64)
    offsets = np.array([-0.24, -0.08, 0.08, 0.24], dtype=np.float64)

    colours = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    markers = ["o", "s", "^", "D"]

    for charge_index, charge_name in enumerate(charge_names):
        axis.errorbar(
            x + offsets[charge_index],
            values[charge_index, :],
            yerr=errors[charge_index, :],
            fmt=markers[charge_index],
            color=colours[charge_index],
            capsize=4,
            markersize=6,
            linestyle="none",
            label=charge_label(charge_name),
        )

    axis.grid(axis="y", alpha=0.30)
    axis.set_xticks(x)
    axis.set_xticklabels([boundary_label(name) for name in boundary_names])
    axis.set_title(title)
    axis.set_xlabel("Boundary condition")
    axis.set_ylabel("")

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_total_potential(rows, grid_size, total_walkers):
    """Save a three-panel total-potential figure with error bars."""

    case_names = get_case_order()
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))

    for axis, case_name in zip(axes, case_names):
        case_rows = get_case_rows(rows, case_name)
        matrices = build_case_matrices(rows, case_name)

        draw_grouped_errorbars(
            axis,
            matrices["total_potential"],
            matrices["total_error"],
            case_title(case_rows),
        )

        axis.set_ylabel("Total potential (V)")

    handles, labels = axes[0].get_legend_handles_labels()

    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.18),
        title="Charge distribution",
    )

    figure.suptitle(
        f"Task 4 total potential comparison (N={grid_size}, walkers={total_walkers}, cython)",
        y=1.04,
    )

    figure.subplots_adjust(top=0.72, wspace=0.28)

    filename = f"task4_total_potential_compare_N{grid_size}_w{total_walkers}x.png"
    figure.savefig(filename, dpi=300, facecolor="white")
    plt.close(figure)

    print(f"Saved {filename}")


def plot_contributions(rows, grid_size, total_walkers):
    """Save a six-panel contribution figure with error bars."""

    case_names = get_case_order()
    figure, axes = plt.subplots(2, 3, figsize=(14.8, 8.2))

    for axis, case_name in zip(axes[0, :], case_names):
        case_rows = get_case_rows(rows, case_name)
        matrices = build_case_matrices(rows, case_name)

        draw_grouped_errorbars(
            axis,
            matrices["boundary_contribution"],
            matrices["boundary_error"],
            case_title(case_rows),
        )

        axis.set_ylabel("Boundary contribution (V)")

    for axis, case_name in zip(axes[1, :], case_names):
        case_rows = get_case_rows(rows, case_name)
        matrices = build_case_matrices(rows, case_name)

        draw_grouped_errorbars(
            axis,
            matrices["charge_contribution"],
            matrices["charge_error"],
            case_title(case_rows),
        )

        axis.set_ylabel("Charge contribution (V)")

    axes[0, 0].text(
        -0.38,
        1.10,
        "Boundary contribution",
        transform=axes[0, 0].transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    axes[1, 0].text(
        -0.38,
        1.10,
        "Charge contribution",
        transform=axes[1, 0].transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()

    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.10),
        title="Charge distribution",
    )

    figure.suptitle(
        f"Task 4 contribution comparison (N={grid_size}, walkers={total_walkers}, cython)",
        y=1.02,
    )

    figure.subplots_adjust(top=0.84, hspace=0.42, wspace=0.28)

    filename = f"task4_contributions_N{grid_size}_w{total_walkers}x.png"
    figure.savefig(filename, dpi=300, facecolor="white")
    plt.close(figure)

    print(f"Saved {filename}")


def save_pivot_summary(rows, grid_size, total_walkers):
    """Save a compact pivot-style CSV for easier report use."""

    filename = f"task4_pivot_summary_N{grid_size}_w{total_walkers}x.csv"

    with open(filename, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)

        for case_name in get_case_order():
            case_rows = get_case_rows(rows, case_name)

            if not case_rows:
                continue

            writer.writerow([case_title(case_rows)])
            writer.writerow(
                [
                    "boundary condition",
                    "charge distribution",
                    "boundary contribution (V)",
                    "charge contribution (V)",
                    "total potential (V)",
                    "boundary error (V)",
                    "charge error (V)",
                    "total error (V)",
                ]
            )

            for boundary_name in get_boundary_order():
                for charge_name in get_charge_order():
                    row = find_record(rows, case_name, boundary_name, charge_name)

                    writer.writerow(
                        [
                            boundary_label(boundary_name),
                            charge_label(charge_name),
                            f"{row['boundary_contribution']:.8e}",
                            f"{row['charge_contribution']:.8e}",
                            f"{row['total_potential']:.8e}",
                            f"{row['boundary_error']:.8e}",
                            f"{row['charge_error']:.8e}",
                            f"{row['total_error']:.8e}",
                        ]
                    )

            writer.writerow([])

    print(f"Saved {filename}")


def main():
    """Load saved Task 4 summaries and generate clearer comparison plots."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="task4_summary_N101_w1000000x.csv",
        help="Task 4 CSV summary written by reconstruct_potential.py",
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
        help="Total number of walkers used in the saved Green's functions.",
    )
    args = parser.parse_args()

    setup_style()
    rows = load_task4_summary(args.input)
    save_pivot_summary(rows, args.grid_size, args.walkers)
    plot_total_potential(rows, args.grid_size, args.walkers)
    plot_contributions(rows, args.grid_size, args.walkers)


if __name__ == "__main__":
    main()
