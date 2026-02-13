"""
Simple test script for Vector3D class.
"""

from Vector3D import Vector3D


def main():
    # Define standard basis vectors
    i = Vector3D(1, 0, 0)
    j = Vector3D(0, 1, 0)
    k = Vector3D(0, 0, 1)

    print("Testing basic operations:\n")

    # Addition
    print("i + j =", i + j)

    # Subtraction
    print("i - j =", i - j)

    # Dot product (should be 0)
    print("i · j =", i.dot(j))

    # Cross product (should be k)
    print("i x j =", i.cross(j))

    # Magnitude test (should be 5)
    v = Vector3D(3, 4, 0)
    print("|(3,4,0)| =", v.magnitude())


if __name__ == "__main__":
    main()
