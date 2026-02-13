"""
vector3d.py

Minimal 3D Cartesian vector class for PH510 Assignment 2.

Implements:
- Initialisation
- String representation
- Magnitude
- Addition and subtraction
- Dot and cross products
"""

import math


class Vector3D:
    """Class representing a 3D Cartesian vector."""

    def __init__(self, x, y, z):
        """Initialise vector with components x, y, z."""
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        """Return formatted string representation."""
        return f"({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"

    def magnitude(self):
        """Return the Euclidean norm |v|."""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def __add__(self, other):
        """Return vector sum self + other."""
        return Vector3D(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
        )

    def __sub__(self, other):
        """Return vector difference self - other."""
        return Vector3D(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
        )

    def dot(self, other):
        """Return dot (scalar) product of two vectors."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        """Return cross (vector) product of two vectors."""
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
