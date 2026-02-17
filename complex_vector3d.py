"""
ComplexVector3D.py

Written for Python 3.12.3

Pylint score: 10

Subclass of Vector3D that supports complex-valued vector components.

Implements the complex dot product:
a · b = a* · b  (where a* = conjugate of the first vector)

Also redfines magnitude to work for complex vectors
"""

import math
from vector3d import Vector3D


class ComplexVector3D(Vector3D):
    """
    Inherits all behaviour from Vector3D but overrides
    the dot product to use complex conjugation.
    """

    def dot(self, other):
        """
        Return the complex dot product using conjugation
        of the first vector components.
        """
        return (
            self.x.conjugate() * other.x +
            self.y.conjugate() * other.y +
            self.z.conjugate() * other.z
        )

    def magnitude(self):
        """
        Return the magnitude of the complex vector using built-in
	math functions
        """
        return math.sqrt(self.dot(self).real)
