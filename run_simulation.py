"""MPI driver script for parallel Ising Metropolis walkers."""

# Import the lattice creation and energy functions.
import ising_model
# Import the compiled Cython Metropolis kernel.
import metropolis_kernel
# Import the MPI communicator tools.
from mpi4py import MPI


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


# Create the global communicator containing all MPI processes.
COMM = MPI.COMM_WORLD
# Get the rank of this process.
RANK = COMM.Get_rank()
# Get the total number of MPI processes.
SIZE = COMM.Get_size()


# Create the initial lattice with random spins for this walker.
lattice = ising_model.create_lattice(LENGTH)

# Print a short header only from rank 0.
if RANK == 0:
    # Report the number of independent walkers being used.
    print("Running", SIZE, "parallel walkers")
    # Report the simulation temperature.
    print("Temperature:", TEMPERATURE)
    # Report the lattice size.
    print("Lattice size:", LENGTH, "x", LENGTH)

# Run the thermalisation period for this walker.
metropolis_kernel.metropolis_sweep(lattice, TEMPERATURE, THERMALISATION_STEPS)

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
global_energy_sum = COMM.reduce(local_average_energy, op=MPI.SUM, root=0)

# Print final combined results only from rank 0.
if RANK == 0:
    # Compute the mean energy across all walkers.
    average_energy = global_energy_sum / SIZE
    # Print the combined average energy.
    print("Average energy:", average_energy)
    # Print the combined average energy per site.
    print("Average energy per site:", average_energy / (LENGTH * LENGTH))
