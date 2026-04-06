"""Task 4: MPI temperature sweep for the 2D XY model."""

# Import a timer for measuring total runtime.
import time

# Import Python's random module for walker-specific seeding.
import random

# Import the XY lattice creation, energy, and correlation functions.
import xy_model

# Import the compiled Cython XY Metropolis kernel.
import xy_metropolis_kernel

# Import the MPI communicator tools.
from mpi4py import MPI

# Import os so the lattice size can be supplied externally.
import os

# Define the lattice size.
LENGTH = int(os.environ.get("LENGTH", "64"))

# Define the minimum temperature in units where J = 1.
TEMPERATURE_MIN = 0.5

# Define the maximum temperature in units where J = 1.
TEMPERATURE_MAX = 1.5

# Define the temperature spacing.
TEMPERATURE_STEP = 0.05

# Define the number of whole lattice updates used for thermalisation.
THERMALISATION_SWEEPS = 200

# Define the number of measurement cycles.
MEASUREMENT_STEPS = 10000

# Define the number of single-site updates between measurements.
SWEEP_STEPS = LENGTH * LENGTH


# Start the total wall-clock timer for the whole script.
start_time = time.perf_counter()

# Create the global communicator containing all MPI processes.
COMM = MPI.COMM_WORLD

# Get the rank of this process.
RANK = COMM.Get_rank()

# Get the total number of MPI processes.
SIZE = COMM.Get_size()

# Build a run-dependent and rank-dependent seed.
seed = time.time_ns() ^ (RANK + 1)

# Seed the Python random number generator for this rank.
random.seed(seed)

# Build the list of temperatures to simulate.
temperatures = []

# Start from the minimum temperature.
temperature = TEMPERATURE_MIN

# Keep adding temperatures until the upper bound is reached.
while temperature <= TEMPERATURE_MAX + 1.0e-12:
    temperatures.append(round(temperature, 10))
    temperature += TEMPERATURE_STEP

# Work out how many correlation distances are measured.
NUMBER_OF_DISTANCES = LENGTH // 2

# Open output file only on rank 0.
if RANK == 0:
    # Build the list of correlation column names.
    correlation_headers = [
        f"Correlation_r{distance}"
        for distance in range(1, NUMBER_OF_DISTANCES + 1)
    ]

    # Open a CSV file for writing results.
    output_file = open(
        f"xy_temperature_sweep_L{LENGTH}_np{SIZE}_ms{MEASUREMENT_STEPS}.csv",
        "w",
    )

    # Write the CSV header.
    output_file.write(
        "Temperature,AverageEnergyPerSite,EnergyErrorPerSite,"
        "SpecificHeatPerSite,"
        + ",".join(correlation_headers)
        + "\n"
    )

    # Print a short terminal header.
    print("Temperature sweep for the 2D XY model")
    print("Number of walkers:", SIZE)
    print("Lattice size:", LENGTH, "x", LENGTH)
    print()

# Loop over all temperatures in the sweep.
for temperature in temperatures:
    # Create a fresh random lattice for this walker at this temperature.
    lattice = xy_model.create_lattice(LENGTH)

    # Run the thermalisation period for this walker.
    xy_metropolis_kernel.metropolis_sweep(
        lattice,
        temperature,
        THERMALISATION_SWEEPS * SWEEPSTEPS,
    )

    # Start the local energy accumulator at zero.
    local_energy_sum = 0.0

    # Start the local squared-energy accumulator at zero.
    local_energy_squared_sum = 0.0

    # Start the local correlation accumulator at zero for each distance.
    local_correlation_sum = [0.0] * NUMBER_OF_DISTANCES

    # Repeat the measurement cycle the required number of times.
    for _ in range(MEASUREMENT_STEPS):
        # Evolve this walker between measurements.
        xy_metropolis_kernel.metropolis_sweep(lattice, temperature, SWEEP_STEPS)

        # Measure the current total energy.
        energy = xy_model.total_energy(lattice)

        # Add this energy to the local sums.
        local_energy_sum += energy
        local_energy_squared_sum += energy * energy

        # Measure the current correlation function.
        correlations = xy_model.correlation_function(lattice)

        # Add this correlation measurement to the local sums.
        for index in range(NUMBER_OF_DISTANCES):
            local_correlation_sum[index] += correlations[index]

    # Compute the local mean energy.
    local_average_energy = local_energy_sum / MEASUREMENT_STEPS

    # Compute the local mean squared energy.
    local_average_energy_squared = (
        local_energy_squared_sum / MEASUREMENT_STEPS
    )

    # Compute the local mean correlation at each distance.
    local_average_correlation = [
        value / MEASUREMENT_STEPS for value in local_correlation_sum
    ]

    # Gather walker averages on rank 0.
    all_average_energies = COMM.gather(local_average_energy, root=0)
    all_average_energies_squared = COMM.gather(
        local_average_energy_squared,
        root=0,
    )
    all_average_correlations = COMM.gather(
        local_average_correlation,
        root=0,
    )

    # Output results on rank 0.
    if RANK == 0:
        # Compute the mean energy across walkers.
        average_energy = sum(all_average_energies) / SIZE

        # Compute the mean squared energy across walkers.
        average_energy_squared = sum(all_average_energies_squared) / SIZE

        # Compute the mean correlation at each distance across walkers.
        average_correlation = []

        for index in range(NUMBER_OF_DISTANCES):
            mean_value = sum(
                walker_values[index] for walker_values in all_average_correlations
            ) / SIZE
            average_correlation.append(mean_value)

        # Work out the number of lattice sites.
        number_of_sites = LENGTH * LENGTH

        # Convert to energy per site.
        average_energy_per_site = average_energy / number_of_sites

        # Compute the specific heat per site from fluctuations.
        specific_heat_per_site = (
            (average_energy_squared - (average_energy * average_energy))
            / (temperature * temperature * number_of_sites)
        )

        # Compute the walker-to-walker variance of the mean energy.
        if SIZE > 1:
            mean_square = sum(
                energy_value * energy_value for energy_value in all_average_energies
            ) / SIZE

            walker_variance = (
                mean_square - (average_energy * average_energy)
            ) * SIZE / (SIZE - 1)

            walker_variance = max(walker_variance, 0.0)

            energy_error = (walker_variance / SIZE) ** 0.5
        else:
            energy_error = 0.0

        # Convert the energy error to an error per site.
        energy_error_per_site = energy_error / number_of_sites

        # Convert the correlation values to CSV text.
        correlation_text = ",".join(str(value) for value in average_correlation)

        # Write this temperature row to the CSV file.
        output_file.write(
            f"{temperature},{average_energy_per_site},{energy_error_per_site},"
            f"{specific_heat_per_site},{correlation_text}\n"
        )

        # Print a short progress line to the terminal.
        print(
            temperature,
            average_energy_per_site,
            energy_error_per_site,
            specific_heat_per_site,
        )

# Close file and print runtime on rank 0.
if RANK == 0:
    output_file.close()

    end_time = time.perf_counter()

    print()
    print("Total runtime (s):", end_time - start_time)
