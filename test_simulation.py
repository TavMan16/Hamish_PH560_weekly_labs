"""Simple test script for the Ising model implementation."""

# Import the Ising model functions from the module
import ising_model


# Set the lattice size for testing
length = 4

# Create a lattice using the model function
lattice = ising_model.create_lattice(length)

# Print the lattice so we can inspect it
print("Lattice:")
for row in lattice:
    print(row)

# Compute the total energy of the lattice
energy = ising_model.total_energy(lattice)

# Print the computed energy
print("Energy:", energy)
