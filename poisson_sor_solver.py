"""Finite-difference Poisson solver using successive over-relaxation (SOR)."""

import numpy as np


class SolverResult:
    """Container for the final potential field and solver diagnostics."""

    def __init__(self, potential, iterations, converged, max_change, omega, grid_spacing):
        self.potential = potential
        self.iterations = iterations
        self.converged = converged
        self.max_change = max_change
        self.omega = omega
        self.grid_spacing = grid_spacing


def validate_grid_inputs(grid_size, grid_spacing, source, boundary_values):
    """Check basic numerical and dimensional consistency of solver inputs."""

    if grid_size < 3:
        raise ValueError("grid_size must be at least 3.")

    if grid_spacing <= 0.0:
        raise ValueError("grid_spacing must be positive.")

    if source.shape != (grid_size, grid_size):
        raise ValueError("source must have shape (grid_size, grid_size).")

    if boundary_values.shape != (grid_size, grid_size):
        raise ValueError("boundary_values must have shape (grid_size, grid_size).")

    if not np.isfinite(source).all():
        raise ValueError("source contains non-finite values.")

    if not np.isfinite(boundary_values).all():
        raise ValueError("boundary_values contains non-finite values.")


def build_boundary_array(grid_size, top, bottom, left, right):
    """Return a grid containing fixed edge potentials."""

    boundary = np.zeros((grid_size, grid_size), dtype=float)

    # Assign the four edges. Interior values remain unused.
    boundary[0, :] = top
    boundary[-1, :] = bottom
    boundary[:, 0] = left
    boundary[:, -1] = right

    # Corners lie on two boundaries; averaging avoids ambiguity.
    boundary[0, 0] = 0.5 * (top + left)
    boundary[0, -1] = 0.5 * (top + right)
    boundary[-1, 0] = 0.5 * (bottom + left)
    boundary[-1, -1] = 0.5 * (bottom + right)

    return boundary


def apply_boundary_conditions(potential, boundary_values):
    """Enforce fixed boundary potentials on the grid edges."""

    potential[0, :] = boundary_values[0, :]
    potential[-1, :] = boundary_values[-1, :]
    potential[:, 0] = boundary_values[:, 0]
    potential[:, -1] = boundary_values[:, -1]


def create_initial_potential(grid_size, boundary_values, initial_guess=None):
    """Create the starting potential field for the iteration."""

    if initial_guess is None:
        potential = np.zeros((grid_size, grid_size), dtype=float)
    else:
        if initial_guess.shape != (grid_size, grid_size):
            raise ValueError("initial_guess must have shape (grid_size, grid_size).")
        potential = initial_guess.astype(float, copy=True)

    apply_boundary_conditions(potential, boundary_values)

    return potential


def compute_optimal_omega(grid_size):
    """Estimate the near-optimal SOR relaxation parameter for a square grid."""

    return 2.0 / (1.0 + np.sin(np.pi / grid_size))


def sor_iteration(potential, source, grid_spacing, omega):
    """
    Perform one SOR sweep over the interior grid.

    Each point is updated using the finite-difference Poisson stencil and
    immediately written back (Gauss-Seidel ordering).
    """

    max_change = 0.0
    spacing_squared = grid_spacing * grid_spacing
    interior_limit = potential.shape[0] - 1

    for row_index in range(1, interior_limit):
        for column_index in range(1, interior_limit):

            old_value = potential[row_index, column_index]

            # Finite-difference form of the Poisson equation
            neighbour_average = (
                potential[row_index + 1, column_index]
                + potential[row_index - 1, column_index]
                + potential[row_index, column_index + 1]
                + potential[row_index, column_index - 1]
                - spacing_squared * source[row_index, column_index]
            ) / 4.0

            new_value = (1.0 - omega) * old_value + omega * neighbour_average
            potential[row_index, column_index] = new_value

            change = abs(new_value - old_value)
            if change > max_change:
                max_change = change

    return max_change


def solve_poisson(
    grid_size,
    grid_spacing,
    source,
    boundary_values,
    tolerance=1.0e-10,
    max_iterations=100000,
    omega=None,
    initial_guess=None,
):
    """
    Solve the 2-D Poisson equation on a square grid using SOR.
    """

    validate_grid_inputs(grid_size, grid_spacing, source, boundary_values)

    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")

    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1.")

    if omega is None:
        omega_value = compute_optimal_omega(grid_size)
    else:
        omega_value = float(omega)

    if omega_value <= 1.0 or omega_value >= 2.0:
        raise ValueError("omega must satisfy 1 < omega < 2.")

    potential = create_initial_potential(grid_size, boundary_values, initial_guess)

    converged = False
    max_change = np.inf

    for iteration_count in range(1, max_iterations + 1):

        max_change = sor_iteration(potential, source, grid_spacing, omega_value)

        apply_boundary_conditions(potential, boundary_values)

        if max_change < tolerance:
            converged = True
            break

    else:
        iteration_count = max_iterations

    return SolverResult(
        potential=potential,
        iterations=iteration_count,
        converged=converged,
        max_change=float(max_change),
        omega=float(omega_value),
        grid_spacing=float(grid_spacing),
    )
