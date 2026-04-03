"""Plot Ising temperature-sweep results for multiple MPI sizes and measurement lengths."""

# Import the CSV module for reading saved results files.
import csv

# Import the operating system tools for checking whether files exist.
import os

# Import matplotlib for plotting the results.
import matplotlib.pyplot as plt


# Define the list of MPI process counts to include.
MPI_PROCESS_COUNTS = [2, 8]

# Define the list of measurement-step counts to include.
MEASUREMENT_STEP_COUNTS = [1000]

# Define the input filename pattern.
INPUT_PATTERN = "ising_temperature_sweep_np{np}_ms{ms}.csv"

# Define the output image file for the energy plot.
ENERGY_PLOT_FILE = "energy_per_site_comparison.png"

# Define the output image file for the specific heat plot.
SPECIFIC_HEAT_PLOT_FILE = "specific_heat_per_site_comparison.png"


# Create a new figure for the energy plot.
plt.figure()

# Loop over all requested MPI process counts.
for process_count in MPI_PROCESS_COUNTS:
    # Loop over all requested measurement-step counts.
    for measurement_steps in MEASUREMENT_STEP_COUNTS:
        # Build the filename for this dataset.
        input_file = INPUT_PATTERN.format(np=process_count, ms=measurement_steps)

        # Skip this dataset if the file does not exist.
        if not os.path.exists(input_file):
            print("Skipping missing file:", input_file)
            continue

        # Create an empty list to store temperatures.
        temperatures = []

        # Create an empty list to store average energies per site.
        average_energies_per_site = []

        # Create an empty list to store energy errors per site.
        energy_errors_per_site = []

        # Open the input CSV file for reading.
        with open(input_file, "r", newline="") as csv_file:
            # Create a CSV dictionary reader to access columns by name.
            reader = csv.DictReader(csv_file)

            # Loop over each row in the CSV file.
            for row in reader:
                # Read the temperature value from the current row.
                temperature = float(row["Temperature"])

                # Read the average energy per site from the current row.
                average_energy_per_site = float(row["AverageEnergyPerSite"])

                # Read the energy error per site from the current row.
                energy_error_per_site = float(row["EnergyErrorPerSite"])

                # Append the temperature to the temperature list.
                temperatures.append(temperature)

                # Append the average energy per site to its list.
                average_energies_per_site.append(average_energy_per_site)

                # Append the energy error per site to its list.
                energy_errors_per_site.append(energy_error_per_site)

        # Build a label describing this dataset.
        label = f"np={process_count}, ms={measurement_steps}"

        # Plot average energy per site against temperature with error bars.
        plt.errorbar(
            temperatures,
            average_energies_per_site,
            yerr=energy_errors_per_site,
            fmt="o-",
            capsize=3,
            markersize=4,
            label=label,
        )

# Label the horizontal axis.
plt.xlabel("Temperature")

# Label the vertical axis.
plt.ylabel("Average energy per site")

# Add a title to the plot.
plt.title("Average energy per site vs temperature")

# Add a grid to make the plot easier to read.
plt.grid(True)

# Add a legend to identify each dataset.
plt.legend()

# Adjust the layout to avoid clipping labels.
plt.tight_layout()

# Save the energy plot to a PNG file.
plt.savefig(ENERGY_PLOT_FILE)


# Create a new figure for the specific heat plot.
plt.figure()

# Loop over all requested MPI process counts.
for process_count in MPI_PROCESS_COUNTS:
    # Loop over all requested measurement-step counts.
    for measurement_steps in MEASUREMENT_STEP_COUNTS:
        # Build the filename for this dataset.
        input_file = INPUT_PATTERN.format(np=process_count, ms=measurement_steps)

        # Skip this dataset if the file does not exist.
        if not os.path.exists(input_file):
            print("Skipping missing file:", input_file)
            continue

        # Create an empty list to store temperatures.
        temperatures = []

        # Create an empty list to store specific heats per site.
        specific_heats_per_site = []

        # Open the input CSV file for reading.
        with open(input_file, "r", newline="") as csv_file:
            # Create a CSV dictionary reader to access columns by name.
            reader = csv.DictReader(csv_file)

            # Loop over each row in the CSV file.
            for row in reader:
                # Read the temperature value from the current row.
                temperature = float(row["Temperature"])

                # Read the specific heat per site from the current row.
                specific_heat_per_site = float(row["SpecificHeatPerSite"])

                # Append the temperature to the temperature list.
                temperatures.append(temperature)

                # Append the specific heat per site to its list.
                specific_heats_per_site.append(specific_heat_per_site)

        # Build a label describing this dataset.
        label = f"np={process_count}, ms={measurement_steps}"

        # Plot specific heat per site against temperature.
        plt.plot(
            temperatures,
            specific_heats_per_site,
            marker="o",
            label=label,
        )

# Label the horizontal axis.
plt.xlabel("Temperature")

# Label the vertical axis.
plt.ylabel("Specific heat per site")

# Add a title to the plot.
plt.title("Specific heat per site vs temperature")

# Add a grid to make the plot easier to read.
plt.grid(True)

# Add a legend to identify each dataset.
plt.legend()

# Adjust the layout to avoid clipping labels.
plt.tight_layout()

# Save the specific heat plot to a PNG file.
plt.savefig(SPECIFIC_HEAT_PLOT_FILE)

# Display both plots on screen.
plt.show()
