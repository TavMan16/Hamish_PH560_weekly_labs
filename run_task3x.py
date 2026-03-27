"""MPI driver for timed Cython Task 3 Green's function runs."""

#pylint score:9.14

# Import command-line argument parsing.
import argparse
# Import NumPy for saving output arrays and timing tables.
import numpy as np
# Import the MPI communicator and wall-clock timer.
from mpi4py import MPI

# Import the parallel Green's function solver.
from poisson_green_mpix import compute_green_function


# Define the three Task 3 starting sites from the assignment.
def get_task3_sites(grid_size):
    """Return the three Task 3 start sites."""

    # Return the centre, near-corner, and face-midpoint sites.
    return {
        "centre": (grid_size // 2, grid_size // 2),
        "corner": (2, 2),
        "face": (grid_size // 2, 2),
    }


# Save the computed Green's function arrays using the existing filename style.
def save_result(result):
    """Save one Green's function result using the existing Cython naming style."""

    # Extract the grid size from the result dictionary.
    grid_size = result["grid_size"]
    # Extract the starting row index.
    start_row = result["start_row"]
    # Extract the starting column index.
    start_column = result["start_column"]
    # Extract the total number of walkers used.
    total_walkers = result["total_walkers"]

    # Build the common filename stem.
    file_stub = f"N{grid_size}_r{start_row}_c{start_column}_w{total_walkers}"

    # Save the charge-related Green's function array.
    np.save(f"green_charge_{file_stub}x.npy", result["charge_green"])
    # Save the charge-related standard deviation array.
    np.save(f"green_charge_std_{file_stub}x.npy", result["charge_std"])
    # Save the boundary-related Green's function array.
    np.save(f"green_boundary_{file_stub}x.npy", result["boundary_green"])
    # Save the boundary-related standard deviation array.
    np.save(f"green_boundary_std_{file_stub}x.npy", result["boundary_std"])

    # Report the saved charge Green's function filename.
    print(f"Saved green_charge_{file_stub}x.npy")
    # Report the saved charge standard deviation filename.
    print(f"Saved green_charge_std_{file_stub}x.npy")
    # Report the saved boundary Green's function filename.
    print(f"Saved green_boundary_{file_stub}x.npy")
    # Report the saved boundary standard deviation filename.
    print(f"Saved green_boundary_std_{file_stub}x.npy")


# Save one timing record in a simple CSV-style text file.
def append_timing_record(case_name, grid_size, total_walkers, comm_size, runtime):
    """Append one timing record for later Amdahl speedup analysis."""

    # Define the output filename for timing data.
    timing_filename = "task3_timings.csv"
    # Define the CSV header line.
    header = "case,grid_size,total_walkers,mpi_processes,parallel_runtime_s\n"
    # Define the CSV data line for this run.
    line = f"{case_name},{grid_size},{total_walkers},{comm_size},{runtime:.12f}\n"

    # Try to read the file first to see whether the header already exists.
    try:
        # Open the timing file in read mode.
        with open(timing_filename, "r", encoding="utf-8") as timing_file:
            # Read the whole file contents.
            existing_contents = timing_file.read()
    # Handle the case where the file does not yet exist.
    except FileNotFoundError:
        # Store an empty string if the file is absent.
        existing_contents = ""

    # Open the timing file in append mode.
    with open(timing_filename, "a", encoding="utf-8") as timing_file:
        # Write the header only if the file was previously empty.
        if not existing_contents:
            # Write the CSV header line.
            timing_file.write(header)
        # Write the timing record for this run.
        timing_file.write(line)


# Run one Task 3 case, time it cleanly, and save outputs on rank 0.
def run_case(
    case_name,
    start_row,
    start_column,
    grid_size,
    grid_spacing,
    total_walkers,
    seed,
    comm,
):
    """Run one Task 3 case and save arrays and timings on rank 0."""

    # Synchronise all ranks before timing begins.
    comm.Barrier()
    # Record the global start time after synchronisation.
    start_time = MPI.Wtime()

    # Execute the parallel Green's function calculation.
    result = compute_green_function(
        start_row,
        start_column,
        grid_size,
        total_walkers,
        grid_spacing=grid_spacing,
        seed=seed,
        comm=comm,
    )

    # Synchronise all ranks again so timing includes the full parallel section.
    comm.Barrier()
    # Record the global end time after the second synchronisation.
    end_time = MPI.Wtime()
    # Compute this rank's elapsed runtime.
    local_runtime = end_time - start_time
    # Reduce to the maximum runtime across ranks, which is the parallel wall time.
    parallel_runtime = comm.reduce(local_runtime, op=MPI.MAX, root=0)

    # Return immediately on non-root ranks.
    if comm.rank != 0:
        return

    # Print a blank line for readability.
    print()
    # Report the completed case name.
    print(f"Finished case: {case_name}")
    # Report the number of MPI processes used.
    print(f"MPI processes: {comm.size}")
    # Report the grid size.
    print(f"Grid size: {result['grid_size']}")
    # Report the starting lattice site.
    print(f"Start site: ({result['start_row']}, {result['start_column']})")
    # Report the total number of walkers.
    print(f"Total walkers: {result['total_walkers']}")
    # Report the measured parallel runtime.
    print(f"Parallel runtime: {parallel_runtime:.6f} s")

    # Save the computed Green's function arrays.
    save_result(result)
    # Append the runtime to the timing CSV file.
    append_timing_record(
        case_name=case_name,
        grid_size=result["grid_size"],
        total_walkers=result["total_walkers"],
        comm_size=comm.size,
        runtime=parallel_runtime,
    )


# Parse command-line options and launch one or more Task 3 calculations.
def main():
    """Configure and run one or all Task 3 Green's function calculations."""

    # Create the argument parser.
    parser = argparse.ArgumentParser()
    # Add the case-selection argument.
    parser.add_argument(
        "--case",
        choices=["centre", "corner", "face", "all"],
        default="all",
        help="Task 3 start site to evaluate.",
    )
    # Add the grid size argument.
    parser.add_argument(
        "--grid-size",
        type=int,
        default=101,
        help="Number of grid points in each direction.",
    )
    # Add the grid spacing argument.
    parser.add_argument(
        "--grid-spacing",
        type=float,
        default=0.01,
        help="Grid spacing in metres.",
    )
    # Add the total walker count argument.
    parser.add_argument(
        "--walkers",
        type=int,
        default=1000000,
        help="Total number of random walkers.",
    )
    # Add the random seed argument.
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Base random seed.",
    )
    # Parse the command-line arguments.
    args = parser.parse_args()

    # Get the global MPI communicator.
    comm = MPI.COMM_WORLD
    # Build the Task 3 starting-site lookup table.
    task3_sites = get_task3_sites(args.grid_size)

    # Choose all three cases if requested.
    if args.case == "all":
        # Define the ordered list of all Task 3 cases.
        case_names = ["centre", "corner", "face"]
    # Otherwise run the single requested case.
    else:
        # Wrap the chosen case in a list for uniform looping.
        case_names = [args.case]

    # Loop over the selected cases.
    for case_name in case_names:
        # Look up the starting row and column for this case.
        start_row, start_column = task3_sites[case_name]
        # Run the selected case with timing and output.
        run_case(
            case_name,
            start_row,
            start_column,
            args.grid_size,
            args.grid_spacing,
            args.walkers,
            args.seed,
            comm,
        )


# Run the main function only when executed as a script.
if __name__ == "__main__":
    # Start the program.
    main()
