"""Task 1: 2D Ising lattice and energy calculation."""

import random


def create_lattice(length):
    """Create an L x L lattice with spins ±1."""
    # Create a list of rows, each containing random spins.
    return [[random.choice([-1, 1]) for _ in range(length)] for _ in range(length)]


def total_energy(lattice):
    """Compute total energy using right and down neighbours only."""
    # Get the lattice size.
    length = len(lattice)

    # Start the total energy at zero.
    energy = 0

    # Loop over every lattice coordinate.
    for i in range(length):
        for j in range(length):
            # Get the spin at the current site.
            spin = lattice[i][j]

            # Get the neighbour to the right, wrapping around at the edge.
            right = lattice[i][(j + 1) % length]

            # Get the neighbour below, wrapping around at the edge.
            down = lattice[(i + 1) % length][j]

            # Add the bond energy for the right neighbour.
            energy += -1 * spin * right

            # Add the bond energy for the lower neighbour.
            energy += -1 * spin * down

    # Return the total energy.
    return energy
