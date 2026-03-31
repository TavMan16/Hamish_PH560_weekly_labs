# Import the C math exponential function for the Metropolis probability.
from libc.math cimport exp

# Import Python's random module for random site selection and acceptance tests.
import random


# Define a function to perform repeated Metropolis spin updates on the lattice.
def metropolis_sweep(lattice, temperature, steps):
    # Declare integer variables for lattice size and loop indices.
    cdef int length = len(lattice)
    cdef int step
    cdef int i
    cdef int j

    # Declare integer variables for the current spin and its four neighbours.
    cdef int spin
    cdef int up
    cdef int down
    cdef int left
    cdef int right

    # Declare the neighbour sum and energy change.
    cdef int neighbour_sum
    cdef int delta_energy

    # Declare the random acceptance number.
    cdef double r

    # Loop over the requested number of single-spin update attempts.
    for step in range(steps):
        # Choose a random row coordinate.
        i = random.randrange(length)

        # Choose a random column coordinate.
        j = random.randrange(length)

        # Read the spin at the selected lattice site.
        spin = lattice[i][j]

        # Read the spin above, wrapping around the boundary.
        up = lattice[(i - 1) % length][j]

        # Read the spin below, wrapping around the boundary.
        down = lattice[(i + 1) % length][j]

        # Read the spin to the left, wrapping around the boundary.
        left = lattice[i][(j - 1) % length]

        # Read the spin to the right, wrapping around the boundary.
        right = lattice[i][(j + 1) % length]

        # Add the four neighbouring spins together.
        neighbour_sum = up + down + left + right

        # Compute the local Metropolis energy change for flipping this spin.
        delta_energy = 2 * spin * neighbour_sum

        # Accept the move immediately if it lowers or preserves the energy.
        if delta_energy <= 0:
            # Flip the spin by changing its sign.
            lattice[i][j] = -spin

        # Otherwise accept the move with Boltzmann probability.
        else:
            # Draw a random number between 0 and 1.
            r = random.random()

            # Accept the move if the random number falls below the Boltzmann factor.
            if r < exp(-delta_energy / temperature):
                # Flip the spin by changing its sign.
                lattice[i][j] = -spin

    # Return the updated lattice after all attempted updates.
    return lattice
