# Import the lattice creation function from the Ising model module.
import ising_model

# Import the compiled Cython Metropolis kernel.
import metropolis_kernel


# Define the lattice size as a module-level constant.
LENGTH = 4

# Define the simulation temperature as a module-level constant.
TEMPERATURE = 2.0

# Define the number of single-spin update attempts.
STEPS = 100


# Create the initial lattice with random spins.
lattice = ising_model.create_lattice(LENGTH)

# Print a heading for the initial lattice.
print("Initial lattice:")

# Print the initial spin configuration row by row.
for row in lattice:
    print(row)

# Print the initial total energy.
print("Initial energy:", ising_model.total_energy(lattice))

# Apply the Metropolis updates using the Cython kernel.
metropolis_kernel.metropolis_sweep(lattice, TEMPERATURE, STEPS)

# Print a heading for the updated lattice.
print("\nUpdated lattice:")

# Print the updated spin configuration row by row.
for row in lattice:
    print(row)

# Print the updated total energy.
print("Updated energy:", ising_model.total_energy(lattice))
