"""Task 3: MPI temperature sweep for the 2D Ising model."""

# Import a timer for measuring total runtime.
import time

#Import da OS
import os

# Import Python's random module for walker-specific seeding.
import random

# Import the lattice creation and energy functions.
import ising_model

# Import the compiled Cython Metropolis kernel.
import metropolis_kernel

# Import the MPI communicator tools.
from mpi4py import MPI


# Define the lattice size.
LENGTH = int(os.environ.get("LENGTH", "4"))

# Define the minimum temperature in units where J = 1.
TEMPERATURE_MIN = 0.5

# Define the maximum temperature in units where J = 1.
TEMPERATURE_MAX = 3.0

# Define the temperature spacing.
TEMPERATURE_STEP = 0.1

# Define the number of whole-lattice updates used for thermalisation.
THERMALISATION_SWEEPS = 10000

# Define the number of measurement cycles.
MEASUREMENT_STEPS = 100000

# Define the number of single-spin updates between measurements.
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

# Open output file only on rank 0.
if RANK == 0:
    # Open a CSV file for writing results.
    output_file = open(
        f"ising_temperature_sweep_L{LENGTH}_np{SIZE}_ms{MEASUREMENT_STEPS}_ts{THERMALISATION_SWEEPS}.csv",
        "w",
    )

    # Write header to file (CSV format).
    output_file.write(
        "Temperature,AverageEnergyPerSite,EnergyErrorPerSite,"
        "SpecificHeatPerSite\n"
    )

    # Print header to terminal.
    print(
        "Temperature AverageEnergyPerSite EnergyErrorPerSite "
        "SpecificHeatPerSite"
    )

# Loop over all temperatures in the sweep.
for temperature in temperatures:
    # Create a fresh random lattice for this walker at this temperature.
    lattice = ising_model.create_lattice(LENGTH)

    # Run the thermalisation period for this walker.
    metropolis_kernel.metropolis_sweep(
    lattice,
    temperature,
    THERMALISATION_SWEEPS * SWEEP_STEPS,
)

    # Start the local energy accumulator at zero.
    local_energy_sum = 0.0

    # Start the local squared-energy accumulator at zero.
    local_energy_squared_sum = 0.0

    # Repeat the measurement cycle the required number of times.
    for _ in range(MEASUREMENT_STEPS):
        metropolis_kernel.metropolis_sweep(lattice, temperature, SWEEP_STEPS)

        energy = ising_model.total_energy(lattice)

        local_energy_sum += energy
        local_energy_squared_sum += energy * energy

    # Compute local averages.
    local_average_energy = local_energy_sum / MEASUREMENT_STEPS
    local_average_energy_squared = local_energy_squared_sum / MEASUREMENT_STEPS

    # Gather walker averages on rank 0.
    all_average_energies = COMM.gather(local_average_energy, root=0)
    all_average_energies_squared = COMM.gather(
        local_average_energy_squared,
        root=0,
    )

    # Output results on rank 0.
    if RANK == 0:
        average_energy = sum(all_average_energies) / SIZE
        average_energy_squared = sum(all_average_energies_squared) / SIZE

        number_of_sites = LENGTH * LENGTH

        average_energy_per_site = average_energy / number_of_sites

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

        # Write to CSV file.
        output_file.write(
            f"{temperature},{average_energy_per_site},{energy_error_per_site},"
            f"{specific_heat_per_site}\n"
        )

        # Print to terminal.
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
