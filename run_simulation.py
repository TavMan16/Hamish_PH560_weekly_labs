"""Simple driver script for the Ising Metropolis simulation."""

# Import the lattice creation and energy functions.
import ising_model

# Import the compiled Cython Metropolis kernel.
import metropolis_kernel


# Define the lattice size.
LENGTH = 8

# Define the simulation temperature.
TEMPERATURE = 0.1

# Define the number of single-spin updates used for thermalisation.
THERMALISATION_STEPS = 10000

# Define the number of measurement cycles.
MEASUREMENT_STEPS = 100000

# Define the number of single-spin updates between measurements.
SWEEP_STEPS = LENGTH * LENGTH


# Create the initial lattice with random spins.
lattice = ising_model.create_lattice(LENGTH)

# Print the starting lattice.
print("Initial lattice:")
for row in lattice:
    print(row)

# Print the starting energy.
print("Initial energy:", ising_model.total_energy(lattice))

# Run the thermalisation period and discard the result history.
metropolis_kernel.metropolis_sweep(lattice, TEMPERATURE, THERMALISATION_STEPS)

# Create an empty list to store measured energies.
energies = []

# Repeat the measurement cycle the required number of times.
for _ in range(MEASUREMENT_STEPS):
    # Evolve the system between measurements.
    metropolis_kernel.metropolis_sweep(lattice, TEMPERATURE, SWEEP_STEPS)

    # Measure the current total energy and store it.
    energies.append(ising_model.total_energy(lattice))

# Compute the average energy from the stored measurements.
average_energy = sum(energies) / len(energies)

# Print the final lattice after the simulation.
print("\nFinal lattice:")
for row in lattice:
    print(row)

# Print the final instantaneous energy.
print("Final energy:", ising_model.total_energy(lattice))

# Print the average measured energy.
print("Average energy:", average_energy)

# Print the average energy per site.
print("Average energy per site:", average_energy / (LENGTH * LENGTH))
