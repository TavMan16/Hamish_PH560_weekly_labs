"""Plot results from the 2D XY temperature sweep."""

# Import the CSV module for reading saved data.
import csv

# Import matplotlib for plotting.
import matplotlib.pyplot as plt


# Define the input CSV file.
FILENAME = "xy_temperature_sweep_L7_np4_ms100000.csv"

# Define the temperatures to show on the correlation plot.
SELECTED_TEMPERATURES = [0.5, 0.8, 1.1, 1.5]

# Define a tolerance for matching floating-point temperatures.
TEMPERATURE_TOLERANCE = 1.0e-9


# Start lists for the scalar observables.
temperatures = []
average_energies = []
energy_errors = []
specific_heats = []

# Start a dictionary for storing correlation curves.
correlation_data = {}


# Open the CSV file for reading.
with open(FILENAME, "r", newline="") as input_file:
    # Create a CSV reader.
    reader = csv.DictReader(input_file)

    # Work out which columns contain correlation data.
    correlation_columns = [
        column_name
        for column_name in reader.fieldnames
        if column_name.startswith("Correlation_r")
    ]

    # Loop over all rows in the CSV file.
    for row in reader:
        # Read the temperature.
        temperature = float(row["Temperature"])

        # Store the scalar observables.
        temperatures.append(temperature)
        average_energies.append(float(row["AverageEnergyPerSite"]))
        energy_errors.append(float(row["EnergyErrorPerSite"]))
        specific_heats.append(float(row["SpecificHeatPerSite"]))

        # Store correlation data for selected temperatures only.
        for chosen_temperature in SELECTED_TEMPERATURES:
            if abs(temperature - chosen_temperature) < TEMPERATURE_TOLERANCE:
                correlation_values = [
                    float(row[column_name]) for column_name in correlation_columns
                ]
                correlation_data[chosen_temperature] = correlation_values


# Work out the fractional lattice lengths for the correlation plot.
number_of_distances = len(correlation_columns)
fractional_lengths = [
    distance / (2 * number_of_distances)
    for distance in range(1, number_of_distances + 1)
]


# Plot average energy per site against temperature.
plt.figure()
plt.errorbar(
    temperatures,
    average_energies,
    yerr=energy_errors,
    marker="o",
    linestyle="-",
)
plt.xlabel("Temperature")
plt.ylabel("Average energy per site")
plt.title("2D XY model: energy per site vs temperature")
plt.grid(True)
plt.tight_layout()


# Plot specific heat per site against temperature.
plt.figure()
plt.plot(
    temperatures,
    specific_heats,
    marker="o",
    linestyle="-",
)
plt.xlabel("Temperature")
plt.ylabel("Specific heat per site")
plt.title("2D XY model: specific heat per site vs temperature")
plt.grid(True)
plt.tight_layout()


# Plot correlation against fractional lattice length.
plt.figure()
for temperature in SELECTED_TEMPERATURES:
    if temperature in correlation_data:
        plt.plot(
            fractional_lengths,
            correlation_data[temperature],
            marker="o",
            linestyle="-",
            label=f"T = {temperature}",
        )

plt.xlabel("Fractional lattice length")
plt.ylabel("Spin correlation")
plt.title("2D XY model: correlation vs fractional lattice length")
plt.grid(True)
plt.legend()
plt.tight_layout()


# Show all plots.
plt.show()
