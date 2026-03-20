"""Plot Task 3 speedup with ideal and fitted Amdahl curves."""

import numpy as np
import matplotlib.pyplot as plt


def load_csv(filename):
    """Load the timing CSV file using NumPy only."""

    data = np.genfromtxt(
        filename,
        delimiter=",",
        names=True,
        dtype=None,
        encoding=None,
    )
    return data


def select_case(data, case_name, grid_size=None, total_walkers=None):
    """Select one Task 3 case and optionally filter by grid size and walkers."""

    mask = data["case"] == case_name

    if grid_size is not None:
        mask = mask & (data["grid_size"] == grid_size)

    if total_walkers is not None:
        mask = mask & (data["total_walkers"] == total_walkers)

    selected = data[mask]

    if selected.size == 0:
        raise ValueError("No matching rows found in task3_timings.csv")

    return selected


def average_repeats(data):
    """Average repeated runs at the same process count."""

    processes = np.unique(data["mpi_processes"])
    mean_runtimes = []

    for process_count in processes:
        mask = data["mpi_processes"] == process_count
        mean_runtimes.append(np.mean(data["parallel_runtime_s"][mask]))

    mean_runtimes = np.array(mean_runtimes)
    return processes, mean_runtimes


def compute_speedup(processes, runtimes):
    """Compute speedup from the 1-process runtime."""

    mask = processes == 1
    if not np.any(mask):
        raise ValueError("A 1-process baseline is required to compute speedup")

    t1 = runtimes[mask][0]
    speedup = t1 / runtimes
    return t1, speedup


def amdahl_speedup(processes, parallel_fraction):
    """Return Amdahl-law speedup for a given parallel fraction."""

    return 1.0 / ((1.0 - parallel_fraction) + (parallel_fraction / processes))


def fit_amdahl(processes, measured_speedup):
    """Fit the Amdahl parallel fraction p by grid search."""

    p_values = np.linspace(0.001, 0.999, 5000)
    best_p = None
    best_error = np.inf

    for parallel_fraction in p_values:
        model_speedup = amdahl_speedup(processes, parallel_fraction)
        error = np.sum((measured_speedup - model_speedup) ** 2)

        if error < best_error:
            best_error = error
            best_p = parallel_fraction

    return best_p, best_error


def inflection_point_log2(parallel_fraction):
    """Return the inflection point location on a log2-processor axis."""

    n_infl = parallel_fraction / (1.0 - parallel_fraction)
    log2_n_infl = np.log2(n_infl)
    return n_infl, log2_n_infl


def plot_speedup(processes, measured_speedup, parallel_fraction, case_name):
    """Plot measured, ideal, and fitted Amdahl speedup curves."""

    smooth_processes = np.linspace(processes.min(), processes.max(), 400)
    fitted_speedup = amdahl_speedup(smooth_processes, parallel_fraction)
    ideal_speedup = smooth_processes

    n_infl, log2_n_infl = inflection_point_log2(parallel_fraction)

    plt.figure(figsize=(7, 5))

    plt.plot(
        processes,
        measured_speedup,
        "o",
        markersize=7,
        label="Measured speedup",
    )

    plt.plot(
        smooth_processes,
        fitted_speedup,
        linewidth=2,
        label=f"Amdahl fit (p = {parallel_fraction:.4f})",
    )

    plt.plot(
        smooth_processes,
        ideal_speedup,
        "--",
        linewidth=1.5,
        label="Ideal speedup",
    )

    if smooth_processes.min() <= n_infl <= smooth_processes.max():
        plt.axvline(
            n_infl,
            color="red",
            linestyle=":",
            linewidth=1.5,
            label=f"Inflection at N = {n_infl:.2f}",
        )

    plt.xscale("log", base=2)
    plt.xlabel("Number of MPI processes")
    plt.ylabel("Speedup")
    plt.title(f"Task 3 speedup for {case_name}")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("task3_amdahl_speedup.png", dpi=300)
    plt.show()

    print(f"Fitted parallel fraction p = {parallel_fraction:.6f}")
    print(f"Inflection point N = p / (1 - p) = {n_infl:.6f}")
    print(f"log2(N_inflection) = {log2_n_infl:.6f}")


def main():
    """Load timings, fit Amdahl's law, and plot the selected Task 3 case."""

    filename = "task3_timings.csv"
    case_name = "centre"
    grid_size = 101
    total_walkers = 1000000

    data = load_csv(filename)
    selected = select_case(
        data,
        case_name=case_name,
        grid_size=grid_size,
        total_walkers=total_walkers,
    )

    processes, runtimes = average_repeats(selected)
    _, measured_speedup = compute_speedup(processes, runtimes)
    parallel_fraction, _ = fit_amdahl(processes, measured_speedup)

    plot_speedup(processes, measured_speedup, parallel_fraction, case_name)


if __name__ == "__main__":
    main()
