"""
geometry_test.py

Written for Python 3.12.3

Pylint score: 10

Compute triangle areas and internal angles for PH510 Assignment 2.
"""

import math
import csv
from vector3d import Vector3D


def triangle_area(a, b, c):
    """Return area of triangle with vertices a, b, c."""
    ab = b - a
    ac = c - a
    return 0.5 * ab.cross(ac).magnitude()


def angle_between(u, v):
    """Return angle between vectors u and v in radians."""
    cos_theta = u.dot(v) / (u.magnitude() * v.magnitude())
    # Clamp to avoid domain errors from floating-point rounding.
    cos_theta = max(-1.0, min(1.0, cos_theta))
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
    """Run a small set of triangle tests and print area/angles."""
    triangles = [
        (Vector3D(0, 0, 0), Vector3D(1, 0, 0), Vector3D(0, 1, 0)),
        (Vector3D(-1, -1, -1), Vector3D(0, -1, -1), Vector3D(-1, 0, -1)),
        (Vector3D(1, 0, 0), Vector3D(0, 0, 1), Vector3D(0, 0, 0)),
        (Vector3D(0, 0, 0), Vector3D(1, -1, 0), Vector3D(0, 0, 1)),
    ]

    # --- Open CSV file for writing ---
    with open("task2_results.csv", "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        # Header row
        writer.writerow([
            "Triangle",
            "Area",
            "Angle A (deg)",
            "Angle B (deg)",
            "Angle C (deg)",
            "Angle Sum (deg)"
        ])

        for i, (a, b, c) in enumerate(triangles, 1):
            area = triangle_area(a, b, c)
            angles = triangle_angles(a, b, c)
            angle_sum = sum(angles)

            # Console output (unchanged)
            print(f"\nTriangle {i}")
            print(f"Area = {area:.6f}")
            print(
                f"Angles (deg) = "
                f"{angles[0]:.6f}, {angles[1]:.6f}, {angles[2]:.6f}"
            )
            print(f"Angle sum = {angle_sum:.6f}")

            # Write to CSV
            writer.writerow([
                i,
                f"{area:.6f}",
                f"{angles[0]:.6f}",
                f"{angles[1]:.6f}",
                f"{angles[2]:.6f}",
                f"{angle_sum:.6f}",
            ])


if __name__ == "__main__":
    main()
