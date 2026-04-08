"""Task 4: MPI temperature sweep for the 2D XY model."""

# Pylint cannot fully inspect symbols provided by mpi4py or compiled
# Cython extension modules, so these warnings are disabled here.
# pylint: disable=no-name-in-module,c-extension-no-member

import os
import random
import time
from contextlib import contextmanager, nullcontext

from mpi4py import MPI

import xy_model
import xy_metropolis_kernel


LENGTH = int(os.environ.get("LENGTH", "8"))
TEMPERATURE_MIN = 0.5
TEMPERATURE_MAX = 3.0
TEMPERATURE_STEP = 0.1
THERMALISATION_SWEEPS = 10000
MEASUREMENT_STEPS = 100000
SWEEP_STEPS = LENGTH * LENGTH
NUMBER_OF_DISTANCES = LENGTH // 2


def build_temperature_list():
    """Return the list of temperatures included in the sweep."""
    temperatures = []
    temperature = TEMPERATURE_MIN

    while temperature <= TEMPERATURE_MAX + 1.0e-12:
        temperatures.append(round(temperature, 10))
        temperature += TEMPERATURE_STEP

    return temperatures


def build_correlation_headers():
    """Return the CSV column names for the correlation measurements."""
    return [
        f"Correlation_r{distance}"
        for distance in range(1, NUMBER_OF_DISTANCES + 1)
    ]


def build_output_filename(size):
    """Return the CSV filename for this XY temperature sweep."""
    return (
        f"xy_temperature_sweep_L{LENGTH}_np{size}"
        f"_ms{MEASUREMENT_STEPS}_ts{THERMALISATION_SWEEPS}.csv"
    )


@contextmanager
def open_output_stream(rank, size):
    """Open the rank-0 output file and yield a file-like object."""
    if rank == 0:
        with open(
            build_output_filename(size),
            "w",
            encoding="utf-8",
        ) as output_file:
            yield output_file
    else:
        with nullcontext() as output_file:
            yield output_file


def run_single_temperature(temperature):
    """Run one temperature point for a single MPI walker."""
    lattice = xy_model.create_lattice(LENGTH)

    xy_metropolis_kernel.metropolis_sweep(
        lattice,
        temperature,
        THERMALISATION_SWEEPS * SWEEP_STEPS,
    )

    local_energy_sum = 0.0
    local_energy_squared_sum = 0.0
    local_correlation_sum = [0.0] * NUMBER_OF_DISTANCES

    for _ in range(MEASUREMENT_STEPS):
        xy_metropolis_kernel.metropolis_sweep(
            lattice,
            temperature,
            SWEEP_STEPS,
        )

        energy = xy_model.total_energy(lattice)
        local_energy_sum += energy
        local_energy_squared_sum += energy * energy

        correlations = xy_model.correlation_function(lattice)
        for index, value in enumerate(correlations[:NUMBER_OF_DISTANCES]):
            local_correlation_sum[index] += value

    return (
        local_energy_sum / MEASUREMENT_STEPS,
        local_energy_squared_sum / MEASUREMENT_STEPS,
        [value / MEASUREMENT_STEPS for value in local_correlation_sum],
    )


def compute_average_correlation(all_average_correlations, size):
    """Return the mean correlation at each distance across walkers."""
    return [
        sum(walker_values[index] for walker_values in all_average_correlations)
        / size
        for index in range(NUMBER_OF_DISTANCES)
    ]


def compute_energy_error(all_average_energies, average_energy, size):
    """Return the walker-to-walker standard error in total energy."""
    if size <= 1:
        return 0.0

    mean_square = sum(
        energy_value * energy_value for energy_value in all_average_energies
    ) / size

    walker_variance = (
        mean_square - (average_energy * average_energy)
    ) * size / (size - 1)

    walker_variance = max(walker_variance, 0.0)
    return (walker_variance / size) ** 0.5


def compute_root_observables(
    temperature,
    all_average_energies,
    all_average_energies_squared,
    size,
):
    """Return averaged energy observables computed on rank 0."""
    number_of_sites = LENGTH * LENGTH
    average_energy = sum(all_average_energies) / size
    average_energy_squared = sum(all_average_energies_squared) / size

    average_energy_per_site = average_energy / number_of_sites
    specific_heat_per_site = (
        (average_energy_squared - (average_energy * average_energy))
        / (temperature * temperature * number_of_sites)
    )

    energy_error_per_site = (
        compute_energy_error(all_average_energies, average_energy, size)
        / number_of_sites
    )

    return average_energy_per_site, energy_error_per_site, specific_heat_per_site


def build_csv_row(
    temperature,
    average_energy_per_site,
    energy_error_per_site,
    specific_heat_per_site,
    average_correlation,
):
    """Return one CSV row for the current temperature."""
    correlation_text = ",".join(str(value) for value in average_correlation)
    return (
        f"{temperature},{average_energy_per_site},"
        f"{energy_error_per_site},{specific_heat_per_site},"
        f"{correlation_text}\n"
    )


def write_header(output_file, size):
    """Write the CSV header and terminal header on rank 0."""
    correlation_headers = ",".join(build_correlation_headers())

    output_file.write(
        "Temperature,AverageEnergyPerSite,EnergyErrorPerSite,"
        f"SpecificHeatPerSite,{correlation_headers}\n"
    )

    print("Temperature sweep for the 2D XY model")
    print("Number of walkers:", size)
    print("Lattice size:", LENGTH, "x", LENGTH)
    print()


def gather_results(comm, local_average_energy, local_average_energy_squared,
local_average_correlation):
    """Gather local walker averages onto rank 0."""
    all_average_energies = comm.gather(local_average_energy, root=0)
    all_average_energies_squared = comm.gather(
        local_average_energy_squared,
        root=0,
    )
    all_average_correlations = comm.gather(
        local_average_correlation,
        root=0,
    )

    return (
        all_average_energies,
        all_average_energies_squared,
        all_average_correlations,
    )


def handle_root_output(
    output_file,
    temperature,
    gathered_results,
    size,
):
    """Compute, write, and print rank-0 results for one temperature."""
    (
        all_average_energies,
        all_average_energies_squared,
        all_average_correlations,
    ) = gathered_results

    average_correlation = compute_average_correlation(
        all_average_correlations,
        size,
    )

    (
        average_energy_per_site,
        energy_error_per_site,
        specific_heat_per_site,
    ) = compute_root_observables(
        temperature,
        all_average_energies,
        all_average_energies_squared,
        size,
    )

    output_file.write(
        build_csv_row(
            temperature,
            average_energy_per_site,
            energy_error_per_site,
            specific_heat_per_site,
            average_correlation,
        )
    )

    print(
        temperature,
        average_energy_per_site,
        energy_error_per_site,
        specific_heat_per_site,
    )


def main():
    """Run the MPI temperature sweep for the 2D XY model."""
    start_time = time.perf_counter()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    random.seed(time.time_ns() ^ (rank + 1))
    temperatures = build_temperature_list()

    with open_output_stream(rank, size) as output_file:
        if rank == 0:
            write_header(output_file, size)

        for temperature in temperatures:
            gathered_results = gather_results(
                comm,
                *run_single_temperature(temperature),
            )

            if rank == 0:
                handle_root_output(
                    output_file,
                    temperature,
                    gathered_results,
                    size,
                )

    if rank == 0:
        end_time = time.perf_counter()
        print()
        print("Total runtime (s):", end_time - start_time)


if __name__ == "__main__":
    main()
