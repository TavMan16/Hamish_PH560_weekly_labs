"""Task 3: MPI temperature sweep for the 2D Ising model."""

# Import a timer for measuring total runtime.
import time

# Import Python's random module for walker-specific seeding.
import random

# Import the lattice creation and energy functions.
import ising_model

# Import the compiled Cython Metropolis kernel.
import metropolis_kernel

# Import the MPI communicator tools.
from mpi4py import MPI


# Define the lattice size.
LENGTH = 7

# Define the minimum temperature in units where J = 1.
TEMPERATURE_MIN = 1.0

# Define the maximum temperature in units where J = 1.
TEMPERATURE_MAX = 3.0

# Define the temperature spacing.
TEMPERATURE_STEP = 0.1

# Define the number of single-spin updates used for thermalisation.
THERMALISATION_STEPS = 1000

# Define the number of measurement cycles.
MEASUREMENT_STEPS = 10000

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

# Give each walker a distinct random seed based on rank.
random.seed(12345 + RANK)

# Build the list of temperatures to simulate.
temperatures = []

# Start from the minimum temperature.
temperature = TEMPERATURE_MIN

# Keep adding temperatures until the upper bound is reached.
while temperature <= TEMPERATURE_MAX + 1.0e-12:
    # Append the current temperature to the list.
    temperatures.append(round(temperature, 10))

    # Increase the temperature by one step.
    temperature += TEMPERATURE_STEP

# Print a short header only from rank 0.
if RANK == 0:
    # Report the number of independent walkers being used.
    print("Running", SIZE, "parallel walkers")

    # Report the lattice size.
    print("Lattice size:", LENGTH, "x", LENGTH)

    # Report the thermalisation length.
    print("Thermalisation steps:", THERMALISATION_STEPS)

    # Report the number of measurement cycles.
    print("Measurement steps:", MEASUREMENT_STEPS)

    # Report the number of spin updates between measurements.
    print("Sweep steps:", SWEEP_STEPS)

    # Print a blank line before the tabulated results.
    print()

    # Print the results table header.
    print("Temperature AverageEnergyPerSite SpecificHeatPerSite")

# Loop over all temperatures in the sweep.
for temperature in temperatures:
    # Create a fresh random lattice for this walker at this temperature.
    lattice = ising_model.create_lattice(LENGTH)

    # Run the thermalisation period for this walker.
    metropolis_kernel.metropolis_sweep(lattice, temperature, THERMALISATION_STEPS)

    # Start the local energy accumulator at zero.
    local_energy_sum = 0.0

    # Start the local squared-energy accumulator at zero.
    local_energy_squared_sum = 0.0

    # Repeat the measurement cycle the required number of times.
    for _ in range(MEASUREMENT_STEPS):
        # Evolve this walker between measurements.
        metropolis_kernel.metropolis_sweep(lattice, temperature, SWEEP_STEPS)

        # Measure the current total energy.
        energy = ising_model.total_energy(lattice)

        # Add the energy to the local sum.
        local_energy_sum += energy

        # Add the squared energy to the local squared-energy sum.
        local_energy_squared_sum += energy * energy

    # Compute the average energy for this walker.
    local_average_energy = local_energy_sum / MEASUREMENT_STEPS

    # Compute the average squared energy for this walker.
    local_average_energy_squared = local_energy_squared_sum / MEASUREMENT_STEPS

    # Reduce all walker average energies onto rank 0 by summing them.
    global_energy_sum = COMM.reduce(local_average_energy, op=MPI.SUM, root=0)

    # Reduce all walker average squared energies onto rank 0 by summing them.
    global_energy_squared_sum = COMM.reduce(
        local_average_energy_squared,
        op=MPI.SUM,
        root=0,
    )

    # Print combined results only from rank 0.
    if RANK == 0:
        # Compute the mean energy across all walkers.
        average_energy = global_energy_sum / SIZE

        # Compute the mean squared energy across all walkers.
        average_energy_squared = global_energy_squared_sum / SIZE

        # Compute the total number of sites in the lattice.
        number_of_sites = LENGTH * LENGTH

        # Compute the average energy per site.
        average_energy_per_site = average_energy / number_of_sites

        # Compute the specific heat per site from energy fluctuations.
        specific_heat_per_site = (
            (average_energy_squared - (average_energy * average_energy))
            / (temperature * temperature * number_of_sites)
        )

        # Print one row of results for this temperature.
        print(
            temperature,
            average_energy_per_site,
            specific_heat_per_site,
        )

# Print the total runtime only from rank 0 after the sweep is complete.
if RANK == 0:
    # Stop the total wall-clock timer for the whole script.
    end_time = time.perf_counter()

    # Print a blank line after the results table.
    print()

    # Print the total runtime of the whole script.
    print("Total runtime (s):", end_time - start_time)
