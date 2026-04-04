"""Plot Ising temperature-sweep results for multiple MPI sizes and measurement lengths."""

import csv
import os
import matplotlib.pyplot as plt


# =========================
# USER INPUT SECTION
# =========================

MPI_PROCESS_COUNTS = [2, 3, 4, 5, 6, 7, 8]
MEASUREMENT_STEP_COUNTS = [100000]

# Toggle error bars (energy plot only)
PLOT_ERROR_BARS = True


# =========================
# FILE NAMING SETUP
# =========================

INPUT_PATTERN = "ising_temperature_sweep_np{np}_ms{ms}.csv"

np_string = "_".join(str(n) for n in MPI_PROCESS_COUNTS)
ms_string = "_".join(str(m) for m in MEASUREMENT_STEP_COUNTS)

ENERGY_PLOT_FILE = f"energy_np{np_string}_ms{ms_string}.png"
SPECIFIC_HEAT_PLOT_FILE = f"specific_heat_np{np_string}_ms{ms_string}.png"

print("Saving plots to:")
print(" ", ENERGY_PLOT_FILE)
print(" ", SPECIFIC_HEAT_PLOT_FILE)


# =========================
# ENERGY PLOT (WITH OPTIONAL ERRORS)
# =========================

plt.figure()

for process_count in MPI_PROCESS_COUNTS:
    for measurement_steps in MEASUREMENT_STEP_COUNTS:

        input_file = INPUT_PATTERN.format(np=process_count, ms=measurement_steps)

        if not os.path.exists(input_file):
            print("Skipping missing file:", input_file)
            continue

        temperatures = []
        energies = []
        errors = []

        with open(input_file, "r", newline="") as csv_file:
            reader = csv.DictReader(csv_file)

            for row in reader:
                temperatures.append(float(row["Temperature"]))
                energies.append(float(row["AverageEnergyPerSite"]))
                errors.append(float(row["EnergyErrorPerSite"]))

        mean_error = sum(errors) / len(errors)

        label = (
            f"{process_count} processes, {measurement_steps} samples "
            f"(mean error = {mean_error:.4g})"
        )

        if PLOT_ERROR_BARS:
            plt.errorbar(
                temperatures,
                energies,
                yerr=errors,
                fmt="o-",
                capsize=3,
                markersize=4,
                label=label,
            )
        else:
            plt.plot(
                temperatures,
                energies,
                marker="o",
                label=label,
            )

plt.xlabel("Temperature")
plt.ylabel("Average energy per site")
plt.title("Average energy per site vs temperature")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(ENERGY_PLOT_FILE)


# =========================
# SPECIFIC HEAT PLOT (NO ERRORS EVER)
# =========================

plt.figure()

for process_count in MPI_PROCESS_COUNTS:
    for measurement_steps in MEASUREMENT_STEP_COUNTS:

        input_file = INPUT_PATTERN.format(np=process_count, ms=measurement_steps)

        if not os.path.exists(input_file):
            continue

        temperatures = []
        specific_heats = []

        with open(input_file, "r", newline="") as csv_file:
            reader = csv.DictReader(csv_file)

            for row in reader:
                temperatures.append(float(row["Temperature"]))
                specific_heats.append(float(row["SpecificHeatPerSite"]))

        # Clean label (no error info needed here)
        label = f"{process_count} processes, {measurement_steps} samples"

        plt.plot(
            temperatures,
            specific_heats,
            marker="o",
            label=label,
        )

plt.xlabel("Temperature")
plt.ylabel("Specific heat per site")
plt.title("Specific heat per site vs temperature")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(SPECIFIC_HEAT_PLOT_FILE)

plt.show()
