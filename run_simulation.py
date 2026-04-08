"""MPI driver script for parallel Ising Metropolis walkers."""

# Pylint cannot fully inspect symbols provided by mpi4py or compiled
# Cython extension modules, so these warnings are disabled here.
# pylint: disable=no-name-in-module,c-extension-no-member

import random
import time

from mpi4py import MPI

import ising_model
import metropolis_kernel


# Define the lattice size.
LENGTH = 7

# Define the simulation temperature.
TEMPERATURE = 2.0

# Define the number of single-spin updates used for thermalisation.
THERMALISATION_STEPS = 1000

# Define the number of measurement cycles.
MEASUREMENT_STEPS = 10000

# Define the number of single-spin updates between measurements.
SWEEP_STEPS = LENGTH * LENGTH


def main():
    """Run the MPI Ising simulation and report averaged energy results."""
    # Start the total wall-clock timer for the whole script.
    start_time = time.perf_counter()

    # Create the global communicator containing all MPI processes.
    comm = MPI.COMM_WORLD

    # Get the rank of this process.
    rank = comm.Get_rank()

    # Get the total number of MPI processes.
    size = comm.Get_size()

    # Build a run-dependent and rank-dependent seed.
    seed = time.time_ns() ^ (rank + 1)

    # Seed the Python random number generator for this rank.
    random.seed(seed)

    # Create the initial lattice with random spins for this walker.
    lattice = ising_model.create_lattice(LENGTH)

    # Print a short header only from rank 0.
    if rank == 0:
        # Report the number of independent walkers being used.
        print("Running", size, "parallel walkers")

        # Report the simulation temperature.
        print("Temperature:", TEMPERATURE)

        # Report the lattice size.
        print("Lattice size:", LENGTH, "x", LENGTH)

        # Print the initial lattice for the rank 0 walker only.
        print("Initial lattice:")
        for row in lattice:
            print(row)

        # Print the initial energy for the rank 0 walker only.
        print("Initial energy:", ising_model.total_energy(lattice))

    # Run the thermalisation period for this walker.
    metropolis_kernel.metropolis_sweep(
        lattice, TEMPERATURE, THERMALISATION_STEPS
    )

    # Start the local energy accumulator at zero.
    local_energy_sum = 0.0

    # Repeat the measurement cycle the required number of times.
    for _ in range(MEASUREMENT_STEPS):
        # Evolve this walker between measurements.
        metropolis_kernel.metropolis_sweep(lattice, TEMPERATURE, SWEEP_STEPS)

        # Measure the current total energy and add it to the local sum.
        local_energy_sum += ising_model.total_energy(lattice)

    # Compute the average energy for this walker.
    local_average_energy = local_energy_sum / MEASUREMENT_STEPS

    # Reduce all walker averages onto rank 0 by summing them.
    global_energy_sum = comm.reduce(local_average_energy, op=MPI.SUM, root=0)

    # Print final combined results only from rank 0.
    if rank == 0:
        # Compute the mean energy across all walkers.
        average_energy = global_energy_sum / size

        # Print the final lattice for the rank 0 walker only.
        print("Final lattice:")
        for row in lattice:
            print(row)

        # Print the final energy for the rank 0 walker only.
        print("Final energy:", ising_model.total_energy(lattice))

        # Print the combined average energy.
        print("Average energy:", average_energy)

        # Print the combined average energy per site.
        print("Average energy per site:", average_energy / (LENGTH * LENGTH))

        # Stop the total wall-clock timer for the whole script.
        end_time = time.perf_counter()

        # Print the total runtime of the whole script.
        print("Total runtime (s):", end_time - start_time)


if __name__ == "__main__":
    main()
