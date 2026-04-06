"""Task 4: Cython Metropolis updates for the 2D XY model."""

import math
import random


# Define the maximum size of a proposed angular change.
ANGLE_STEP = math.pi / 4.0


def metropolis_sweep(lattice, temperature, steps):
    """Perform Metropolis updates on an XY lattice."""
    # Get the lattice size.
    cdef int length = len(lattice)

    # Declare loop and index variables.
    cdef int step, i, j

    # Declare neighbour indices.
    cdef int up, down, left, right

    # Declare angle and energy variables.
    cdef double old_theta, new_theta
    cdef double theta_up, theta_down, theta_left, theta_right
    cdef double old_energy, new_energy, delta_energy
    cdef double angle_change

    # Repeat the requested number of single-site update attempts.
    for step in range(steps):
        # Choose a random lattice site.
        i = random.randrange(length)
        j = random.randrange(length)

        # Get the current angle at this site.
        old_theta = lattice[i][j]

        # Propose a small random angular change.
        angle_change = random.uniform(-ANGLE_STEP, ANGLE_STEP)
        new_theta = old_theta + angle_change

        # Wrap the proposed angle into the range [0, 2*pi).
        new_theta = new_theta % (2.0 * math.pi)

        # Work out the neighbour indices with periodic boundaries.
        up = (i - 1) % length
        down = (i + 1) % length
        left = (j - 1) % length
        right = (j + 1) % length

        # Get the neighbouring angles.
        theta_up = lattice[up][j]
        theta_down = lattice[down][j]
        theta_left = lattice[i][left]
        theta_right = lattice[i][right]

        # Compute the old local energy contribution of this site.
        old_energy = 0.0
        old_energy += -math.cos(old_theta - theta_up)
        old_energy += -math.cos(old_theta - theta_down)
        old_energy += -math.cos(old_theta - theta_left)
        old_energy += -math.cos(old_theta - theta_right)

        # Compute the proposed new local energy contribution.
        new_energy = 0.0
        new_energy += -math.cos(new_theta - theta_up)
        new_energy += -math.cos(new_theta - theta_down)
        new_energy += -math.cos(new_theta - theta_left)
        new_energy += -math.cos(new_theta - theta_right)

        # Compute the change in energy for the proposed move.
        delta_energy = new_energy - old_energy

        # Accept all energy-lowering moves.
        if delta_energy <= 0.0:
            lattice[i][j] = new_theta

        # Otherwise accept with the Metropolis probability.
        elif random.random() < math.exp(-delta_energy / temperature):
            lattice[i][j] = new_theta
