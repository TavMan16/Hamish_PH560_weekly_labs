"""Task 4: 2D XY lattice, energy calculation, and spin correlation."""

import math
import random


def create_lattice(length):
    """Create an L x L lattice of random spin angles in radians."""
    # Create a list of rows, each containing a random angle in [0, 2*pi).
    return [
        [random.uniform(0.0, 2.0 * math.pi) for _ in range(length)]
        for _ in range(length)
    ]


def total_energy(lattice):
    """Compute total XY energy using right and down neighbours only."""
    # Get the lattice size.
    length = len(lattice)

    # Start the total energy at zero.
    energy = 0.0

    # Loop over every lattice coordinate.
    for i in range(length):
        for j in range(length):
            # Get the angle at the current site.
            theta = lattice[i][j]

            # Get the neighbour to the right, wrapping around at the edge.
            right = lattice[i][(j + 1) % length]

            # Get the neighbour below, wrapping around at the edge.
            down = lattice[(i + 1) % length][j]

            # Add the bond energy for the right neighbour.
            energy += -1.0 * math.cos(theta - right)

            # Add the bond energy for the lower neighbour.
            energy += -1.0 * math.cos(theta - down)

    # Return the total energy.
    return energy


def correlation_function(lattice):
    """Compute spin correlation C(r) as a function of lattice separation."""
    # Get the lattice size.
    length = len(lattice)

    # Only measure unique separations up to half the lattice length.
    maximum_distance = length // 2

    # Start a list to store the correlation at each separation.
    correlations = []

    # Loop over all separations.
    for distance in range(1, maximum_distance + 1):
        # Start the correlation sum at zero.
        correlation_sum = 0.0

        # Count the number of terms included in the average.
        count = 0

        # Loop over every lattice site.
        for i in range(length):
            for j in range(length):
                # Get the reference angle.
                theta = lattice[i][j]

                # Get the angle distance sites to the right.
                theta_x = lattice[i][(j + distance) % length]

                # Get the angle distance sites downward.
                theta_y = lattice[(i + distance) % length][j]

                # Add both horizontal and vertical correlations.
                correlation_sum += math.cos(theta - theta_x)
                correlation_sum += math.cos(theta - theta_y)

                # Count both contributions.
                count += 2

        # Store the average correlation for this separation.
        correlations.append(correlation_sum / count)

    # Return the list of correlations for r = 1, 2, ..., L//2.
    return correlations
