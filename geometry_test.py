"""
geometry_test.py

Compute triangle areas and internal angles for PH510 Assignment 2.
"""

import math
from Vector3D import Vector3D


def triangle_area(a, b, c):
    """Return area of triangle with vertices a, b, c."""
    ab = b - a
    ac = c - a
    return 0.5 * ab.cross(ac).magnitude()


def angle_between(u, v):
    """Return angle between vectors u and v in radians."""
    cos_theta = u.dot(v) / (u.magnitude() * v.magnitude())
    cos_theta = max(-1.0, min(1.0, cos_theta))  # Clamp to avoid acos domain errors from floating-point rounding
    return math.acos(cos_theta)


def triangle_angles(a, b, c):
    """Return the three internal angles (in degrees)."""
    ab = b - a
    ac = c - a
    ba = a - b
    bc = c - b
    ca = a - c
    cb = b - c

    angle_a = math.degrees(angle_between(ab, ac))
    angle_b = math.degrees(angle_between(ba, bc))
    angle_c = math.degrees(angle_between(ca, cb))

    return angle_a, angle_b, angle_c


def main():
    triangles = [
        (Vector3D(0, 0, 0), Vector3D(1, 0, 0), Vector3D(0, 1, 0)),
        (Vector3D(-1, -1, -1), Vector3D(0, -1, -1), Vector3D(-1, 0, -1)),
        (Vector3D(1, 0, 0), Vector3D(0, 0, 1), Vector3D(0, 0, 0)),
        (Vector3D(0, 0, 0), Vector3D(1, -1, 0), Vector3D(0, 0, 1)),
    ]

    for i, (a, b, c) in enumerate(triangles, 1):
        area = triangle_area(a, b, c)
        angles = triangle_angles(a, b, c)

        print(f"\nTriangle {i}")
        print(f"Area = {area:.6f}")
        print(f"Angles (deg) = {angles[0]:.6f}, {angles[1]:.6f}, {angles[2]:.6f}")
        print(f"Angle sum = {sum(angles):.6f}")


if __name__ == "__main__":
    main()
