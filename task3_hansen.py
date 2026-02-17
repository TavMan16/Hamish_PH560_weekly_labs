"""
task3_hansen.py

Written for Python 3.12.3

Pylint score: 9.87

Task 3 (Assignment 2): numerical verification of the “Hansen vector” identities.

Fields are defined as given in the assignment:
    k = pi*(0,0,1)
    M(x,y,z) = (1,0,0) * exp(i*pi*z)
    N(x,y,z) = (0,1,0) * exp(i*pi*z)

Using central finite differences, we compute:
    - div(M), div(N)
    - curl(M), curl(N)

We then compare the numerical results against TWO sets of relations:

(A) The relations stated in the assignment:
        div(M) = 0
        div(N) = 0
        curl(N) = M/|k|
        curl(M) = N/|k|
    Expected outcome for these specific M,N:
        - divergence checks should be (near) zero
        - curl checks typically do NOT go to zero (a scaling/phase mismatch appears)

(B) The analytic curl relations implied directly by differentiating exp(i*pi*z):
        curl(N) = -i|k| * M
        curl(M) =  +i|k| * N
    Expected outcome:
        - the residuals for (B) should be (near) zero, confirming the numerical curl is correct.

The script prints residuals and PASS/FAIL messages at a small set of test points.
"""

import cmath
from complex_vector3d import ComplexVector3D  # My complex vector class.


K_MAGNITUDE = float(cmath.pi)  # |k| for k = pi*(0,0,1).


def hansen_m(_x, _y, z):
    """Return M(x,y,z) = (1,0,0) * exp(i*pi*z)."""
    phase = cmath.exp(1j * cmath.pi * z)
    return ComplexVector3D(phase, 0j, 0j)


def hansen_n(_x, _y, z):
    """Return N(x,y,z) = (0,1,0) * exp(i*pi*z)."""
    phase = cmath.exp(1j * cmath.pi * z)
    return ComplexVector3D(0j, phase, 0j)


def partial_x(field_func, x_val, y_val, z_val, step):
    """Central-difference approximation to ∂field_func/∂x."""
    f_plus = field_func(x_val + step, y_val, z_val)
    f_minus = field_func(x_val - step, y_val, z_val)
    scale = 1.0 / (2.0 * step)
    return (f_plus - f_minus) * scale


def partial_y(field_func, x_val, y_val, z_val, step):
    """Central-difference approximation to ∂field_func/∂y."""
    f_plus = field_func(x_val, y_val + step, z_val)
    f_minus = field_func(x_val, y_val - step, z_val)
    scale = 1.0 / (2.0 * step)
    return (f_plus - f_minus) * scale


def partial_z(field_func, x_val, y_val, z_val, step):
    """Central-difference approximation to ∂field_func/∂z."""
    f_plus = field_func(x_val, y_val, z_val + step)
    f_minus = field_func(x_val, y_val, z_val - step)
    scale = 1.0 / (2.0 * step)
    return (f_plus - f_minus) * scale


def divergence(field_func, x_val, y_val, z_val, step):
    """Compute div(field_func) at (x_val,y_val,z_val) by finite differences."""
    dfdx = partial_x(field_func, x_val, y_val, z_val, step)
    dfdy = partial_y(field_func, x_val, y_val, z_val, step)
    dfdz = partial_z(field_func, x_val, y_val, z_val, step)
    return dfdx.x + dfdy.y + dfdz.z


def curl(field_func, x_val, y_val, z_val, step):
    """Compute curl(field_func) at (x_val,y_val,z_val) by finite differences."""
    dfdx = partial_x(field_func, x_val, y_val, z_val, step)
    dfdy = partial_y(field_func, x_val, y_val, z_val, step)
    dfdz = partial_z(field_func, x_val, y_val, z_val, step)

    curl_x = dfdy.z - dfdz.y
    curl_y = dfdz.x - dfdx.z
    curl_z = dfdx.y - dfdy.x

    return ComplexVector3D(curl_x, curl_y, curl_z)


def is_zero_complex(value):
    """Return True if a complex number is exactly 0j."""
    return value == 0j


def is_zero_vector(vec):
    """Return True if a ComplexVector3D is exactly the zero vector."""
    return vec.x == 0j and vec.y == 0j and vec.z == 0j


def main():
    """Run the Hansen identity checks and print pass/fail messages."""
    h_step = 1.0e-6

    test_points = [
        (0.0, 0.0, 0.0),
        (0.2, 0.3, 0.4),
        (1.0, 1.0, 1.0),
        (68.0, 69.6, 70.0),
    ]

    for (x0, y0, z0) in test_points:
        print("\nTesting at point:", (x0, y0, z0))

        div_m_val = divergence(hansen_m, x0, y0, z0, h_step)
        div_n_val = divergence(hansen_n, x0, y0, z0, h_step)

        curl_n_val = curl(hansen_n, x0, y0, z0, h_step)
        curl_m_val = curl(hansen_m, x0, y0, z0, h_step)

        # Assignment-stated RHS checks:
        rhs_for_n_assign = hansen_m(x0, y0, z0) * (1.0 / K_MAGNITUDE)
        rhs_for_m_assign = hansen_n(x0, y0, z0) * (1.0 / K_MAGNITUDE)

        # Analytic RHS checks for these M and N:
        rhs_for_n_analytic = (-1j * K_MAGNITUDE) * hansen_m(x0, y0, z0)
        rhs_for_m_analytic = (1j * K_MAGNITUDE) * hansen_n(x0, y0, z0)

        res_div_m = div_m_val
        res_div_n = div_n_val

        res_curl_n_assign = curl_n_val - rhs_for_n_assign
        res_curl_m_assign = curl_m_val - rhs_for_m_assign

        res_curl_n_analytic = curl_n_val - rhs_for_n_analytic
        res_curl_m_analytic = curl_m_val - rhs_for_m_analytic

        print("div M =", div_m_val)
        print("div N =", div_n_val)

        print("curl N - (M/|k|) =", res_curl_n_assign)
        print("curl M - (N/|k|) =", res_curl_m_assign)

        print("curl N - (-i|k| M) =", res_curl_n_analytic)
        print("curl M - ( i|k| N) =", res_curl_m_analytic)

        # ---- PASS / FAIL messages ----

        if is_zero_complex(res_div_m) and is_zero_complex(res_div_n):
            print("PASS: divergence conditions satisfied (div M = 0 and div N = 0).")
        else:
            print("FAIL: divergence conditions NOT satisfied.")

        if is_zero_vector(res_curl_n_assign) and is_zero_vector(res_curl_m_assign):
            print("PASS: assignment curl conditions satisfied.")
        else:
            print("FAIL: assignment curl conditions NOT satisfied.")

        if is_zero_vector(res_curl_n_analytic) and is_zero_vector(res_curl_m_analytic):
            print("PASS: analytic curl conditions satisfied for these M and N.")
        else:
            print("FAIL: analytic curl conditions NOT satisfied for these M and N.")


if __name__ == "__main__":
    main()
