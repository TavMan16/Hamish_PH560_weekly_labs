"""
Vector3D.py

Written for Python 3.12.3

Pylint score: 10

3D Cartesian vector class for PH510 Assignment 2.

Implements:
- Initialisation
- String representation
- Magnitude
- Addition and subtraction
- Dot and cross products
- Mulitplication and division by scalars
"""

import math  # Provides sqrt for the magnitude calculation.


class Vector3D:
    """Class representing a 3D Cartesian vector."""

    def __init__(self, x, y, z):
        """Initialise vector with components x, y, z."""
        self.x = x  # Store x-component.
        self.y = y  # Store y-component.
        self.z = z  # Store z-component.

    def __str__(self):
        """Return formatted string representation."""
        return f"({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"  # Format to 2 d.p.

    def magnitude(self):
        """Return the Euclidean norm |v| for real-valued vectors."""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)  # sqrt(x^2+y^2+z^2)

    def __add__(self, other):
        """Return vector sum self + other."""
        return type(self)(  # Preserve subclass type (e.g., ComplexVector3D).
            self.x + other.x,  # Add x-components.
            self.y + other.y,  # Add y-components.
            self.z + other.z,  # Add z-components.
        )

    def __sub__(self, other):
        """Return vector difference self - other."""
        return type(self)(  # Preserve subclass type (e.g., ComplexVector3D).
            self.x - other.x,  # Subtract x-components.
            self.y - other.y,  # Subtract y-components.
            self.z - other.z,  # Subtract z-components.
        )

    def __mul__(self, scalar):
        """Return vector multiplied by a scalar (vector * scalar)."""
        return type(self)(  # Preserve subclass type (e.g., ComplexVector3D).
            self.x * scalar,  # Scale x-component.
            self.y * scalar,  # Scale y-component.
            self.z * scalar,  # Scale z-component.
        )

    def __rmul__(self, scalar):
        """Return scalar multiplied by vector (scalar * vector)."""
        return self.__mul__(scalar)  # Reuse __mul__ implementation.

    def __truediv__(self, scalar):
        """Return vector divided by a scalar (vector / scalar)."""
        return type(self)(  # Preserve subclass type (e.g., ComplexVector3D).
            self.x / scalar,  # Divide x-component.
            self.y / scalar,  # Divide y-component.
            self.z / scalar,  # Divide z-component.
        )

    def dot(self, other):
        """Return dot product of two vectors."""
        return self.x * other.x + self.y * other.y + self.z * other.z  # Standard dot product.

    def cross(self, other):
        """Return cross (vector) product of two vectors."""
        return type(self)(  # Preserve subclass type (e.g., ComplexVector3D).
            self.y * other.z - self.z * other.y,  # x-component.
            self.z * other.x - self.x * other.z,  # y-component.
            self.x * other.y - self.y * other.x,  # z-component.
        )
