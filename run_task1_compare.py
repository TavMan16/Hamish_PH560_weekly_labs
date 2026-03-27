"""Run deterministic Task 1 comparison cases and save a CSV for Task 5."""

#pylint score:9.36

# import the standard csv module so results can be written to a CSV file
import csv

# import NumPy for array creation and numerical work
import numpy as np

# import the deterministic Poisson solver and boundary helper from your solver module
from poisson_sor_solver import solve_poisson, build_boundary_array

# define the grid size to match the Task 4 Green's-function calculations
GRID_SIZE = 101

# define the grid spacing in metres to match the 1 m square domain with 101 points
GRID_SPACING = 0.01

# define the solver tolerance for the deterministic relaxation method
TOLERANCE = 1.0e-8

# define the maximum number of SOR iterations allowed
MAX_ITERATIONS = 20000

# define the three evaluation points required for comparison with Task 4
POINTS = {
    "centre": (50, 50),
    "face": (50, 2),
    "corner": (2, 2),
}

# define the boundary-condition cases so their names match the Task 4 CSV file
BOUNDARY_CASES = {
    "uniform_100": {
        "top": 100.0,
        "bottom": 100.0,
        "left": 100.0,
        "right": 100.0,
    },
    "tb_100_lr_minus100": {
        "top": 100.0,
        "bottom": 100.0,
        "left": -100.0,
        "right": -100.0,
    },
    "tl_200_b_0_r_minus400": {
        "top": 200.0,
        "bottom": 0.0,
        "left": 200.0,
        "right": -400.0,
    },
}

# define the charge-case names so they match the Task 4 CSV file
CHARGE_CASES = (
    "zero",
    "uniform_10",
    "top_to_bottom_gradient",
    "exp_centre",
)

# define a helper to build the zero-source case
def make_zero_source():
    """Return a source array for the zero-charge case."""

    # create a grid-sized source array filled with zeros
    source = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)

    # return the completed source array
    return source

# define a helper to build the uniform source case
def make_uniform_10_source():
    """Return a source array for the uniform_10 charge case."""

    # create a grid-sized source array filled with a constant value of 10
    source = np.full((GRID_SIZE, GRID_SIZE), 10.0, dtype=float)

    # return the completed source array
    return source

# define a helper to build the top-to-bottom gradient source case
def make_top_to_bottom_gradient_source():
    """Return a source array for the top_to_bottom_gradient charge case."""

    # create an empty grid-sized source array
    source = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)

    # loop over every row index in the grid
    for row_index in range(GRID_SIZE):
        # assign a linearly varying value to the whole row
        source[row_index, :] = row_index / (GRID_SIZE - 1)

    # return the completed source array
    return source

# define a helper to build the centred exponential source case
def make_exp_centre_source():
    """Return a source array for the exp_centre charge case."""

    # create an empty grid-sized source array
    source = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)

    # define the centre row index of the grid
    centre_row = GRID_SIZE // 2

    # define the centre column index of the grid
    centre_column = GRID_SIZE // 2

    # loop over every row index in the grid
    for row_index in range(GRID_SIZE):
        # loop over every column index in the grid
        for column_index in range(GRID_SIZE):
            # calculate the squared distance from the centre in grid units
            radius_squared = (row_index - centre_row) ** 2 + (column_index - centre_column) ** 2

            # assign the exponentially decaying source value at this grid point
            source[row_index, column_index] = np.exp(-radius_squared / 200.0)

    # return the completed source array
    return source

# define a dispatcher that chooses the correct source array for a named charge case
def build_source_array(charge_name):
    """Build and return the source array for one named charge case."""

    # check whether the chosen charge case is zero
    if charge_name == "zero":
        # return the zero-charge source array
        return make_zero_source()

    # check whether the chosen charge case is the uniform_10 case
    if charge_name == "uniform_10":
        # return the uniform source array
        return make_uniform_10_source()

    # check whether the chosen charge case is the top-to-bottom gradient case
    if charge_name == "top_to_bottom_gradient":
        # return the gradient source array
        return make_top_to_bottom_gradient_source()

    # check whether the chosen charge case is the centred exponential case
    if charge_name == "exp_centre":
        # return the exponential source array
        return make_exp_centre_source()

    # raise an error if an unknown charge case name is supplied
    raise ValueError(f"Unknown charge case: {charge_name}")

# define a helper that builds the boundary array for one named boundary case
def build_boundary_values(boundary_name):
    """Build and return the boundary array for one named boundary case."""

    # look up the voltage values for the requested boundary case
    boundary_case = BOUNDARY_CASES[boundary_name]

    # build the full boundary array using the helper from poisson_sor_solver.py
    boundary_values = build_boundary_array(
        GRID_SIZE,
        boundary_case["top"],
        boundary_case["bottom"],
        boundary_case["left"],
        boundary_case["right"],
    )

    # return the completed boundary array
    return boundary_values

# define a helper that solves one deterministic case
def solve_one_case(boundary_name, charge_name):
    """Solve one deterministic Poisson problem for one boundary and charge case."""

    # build the source array for the selected charge case
    source = build_source_array(charge_name)

    # build the boundary array for the selected boundary case
    boundary_values = build_boundary_values(boundary_name)

    # run the deterministic SOR Poisson solver for this case
    result = solve_poisson(
        GRID_SIZE,
        GRID_SPACING,
        source,
        boundary_values,
        tolerance=TOLERANCE,
        max_iterations=MAX_ITERATIONS,
    )

    # return the full solver result object
    return result

# define a helper that converts one solver result into CSV rows for the comparison points
def rows_for_result(boundary_name, charge_name, result):
    """Create CSV rows for the centre, face, and corner comparison points."""

    # create an empty list that will collect one row per comparison point
    rows = []

    # loop over every named comparison point
    for case_name, (row_index, column_index) in POINTS.items():
        # calculate the physical x coordinate in metres from the column index
        x_m = column_index * GRID_SPACING

        # calculate the physical y coordinate in metres from the row index
        y_m = row_index * GRID_SPACING

        # extract the deterministic potential at the requested grid point
        total_potential = float(result.potential[row_index, column_index])

        # build one CSV row matching the Task 4 naming style
        row = {
            "case": case_name,
            "start_row": row_index,
            "start_column": column_index,
            "x_m": x_m,
            "y_m": y_m,
            "x_cm": 100.0 * x_m,
            "y_cm": 100.0 * y_m,
            "boundary_name": boundary_name,
            "charge_name": charge_name,
            "total_potential": total_potential,
            "converged": bool(result.converged),
            "iterations": int(result.iterations),
            "final_max_update": float(result.max_change),
            "omega": float(result.omega),
        }

        # append the completed row to the output list
        rows.append(row)

    # return the list of rows for this solved case
    return rows

# define a helper that writes all rows to a CSV file
def write_results_csv(rows, output_filename):
    """Write the deterministic comparison data to a CSV file."""

    # define the output column order explicitly
    fieldnames = [
        "case",
        "start_row",
        "start_column",
        "x_m",
        "y_m",
        "x_cm",
        "y_cm",
        "boundary_name",
        "charge_name",
        "total_potential",
        "converged",
        "iterations",
        "final_max_update",
        "omega",
    ]

    # open the output file for writing
    with open(output_filename, "w", newline="", encoding="utf-8") as handle:
        # create a dictionary-based CSV writer with the chosen field order
        writer = csv.DictWriter(handle, fieldnames=fieldnames)

        # write the header row to the CSV file
        writer.writeheader()

        # write all data rows to the CSV file
        writer.writerows(rows)

# define the main program entry point
def main():
    """Run all deterministic Task 1 comparison cases and save the output CSV."""

    # create an empty list to collect all CSV rows from all solved cases
    all_rows = []

    # loop over each named boundary case
    for boundary_name in BOUNDARY_CASES:
        # loop over each named charge case
        for charge_name in CHARGE_CASES:
            # print a progress message so the batch output shows activity
            print(f"Running boundary={boundary_name}, charge={charge_name}")

            # solve the current deterministic case
            result = solve_one_case(boundary_name, charge_name)

            # print a short convergence summary for this case
            print(
                f"  converged={result.converged}, "
                f"iterations={result.iterations}, "
                f"max_change={result.max_change:.3e}, "
                f"omega={result.omega:.4f}"
            )

            # convert the solver result into rows for the comparison points
            case_rows = rows_for_result(boundary_name, charge_name, result)

            # extend the master list with those rows
            all_rows.extend(case_rows)

    # define the output CSV filename expected by the comparison script
    output_filename = f"task1_summary_N{GRID_SIZE}.csv"

    # write the full deterministic dataset to the output CSV file
    write_results_csv(all_rows, output_filename)

    # print a final success message with the output filename
    print(f"Saved deterministic comparison data to {output_filename}")

# run the main function only when this script is executed directly
if __name__ == "__main__":
    # call the main driver function
    main()
