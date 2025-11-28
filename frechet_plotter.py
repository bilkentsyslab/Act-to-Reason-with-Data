"""
Before running, set your study_name and training_or_simulation variables accordingly.
This script reads Frechet distance data from a CSV file, filters out zero distances
and plots both the raw Frechet distances and their running average over episodes.
"""

import matplotlib.pyplot as plt

def plot_point_distances(point_distances, window_size=10):
    # Filter out rows with 0.0 distances
    filtered_distances = [dist for dist in point_distances if dist > 0.0]

    # Calculate running average
    running_avg = [
        sum(filtered_distances[max(0, i - window_size):i]) / min(i, window_size)
        for i in range(1, len(filtered_distances) + 1)
    ]

    # Plot both raw Frechet distances and the running average
    plt.figure(figsize=(10, 5))
    plt.plot(filtered_distances, color="blue", label="Frechet Distance (Filtered)")
    plt.plot(running_avg, color="red", label=f"Running Average")
    plt.title("Point Distance vs. Episode")
    plt.xlabel("Episode")
    plt.ylabel("Frechet Distance")
    plt.legend()
    plt.grid(True)
    plt.savefig("frechet_distance_vs_episode.png")
    plt.show()

# Example Usage
import pandas as pd

# Read the CSV file
study_name = "study_data_calib"
training_or_simulation = "training"  # or "simulation"
csv_path = f"NGSIM_I80_Results/{study_name}/{training_or_simulation}/frechet_histories/{training_or_simulation}_frechet_distances.csv"
df = pd.read_csv(csv_path)

# Extract the "Distance" column as a list
point_distances = df["Point Distance"].tolist()

# Call the updated function
plot_point_distances(point_distances, window_size=100)