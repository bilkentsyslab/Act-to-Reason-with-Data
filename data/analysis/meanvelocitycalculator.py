import numpy as np
import pandas as pd

def calculate_velocity_stats(filename):
    # Load the CSV file
    data = pd.read_csv(filename)

    # Extract relevant columns
    data["lane"] = data["Lane_ID"]
    data["x"] = data["Local_Y"] * 0.3048  # Convert feet to meters
    data["v"] = data["Mean_Speed"] * 0.3048  # Convert speed to meters

    min_x = np.min(data["x"])
    data["x"] = data["x"] - min_x

    # Get the merging region end point
    offset = 0 #50m if you dont include the minx code above
    merge_end = 213.66 + offset  # Assuming the value from your results

    # Define road length as merge_end + 50
    road_length = merge_end + 50

    # Filter out data beyond the road length and in lanes 1 to 5
    filtered_data = data[(offset <= data["x"]) & (data["x"] <= road_length) & (data["lane"] >= 6) & (data["lane"] <= 7)]

    # Calculate mean and standard deviation of velocity
    velocity_mean = filtered_data["v"].mean()
    velocity_std = filtered_data["v"].std()

    return velocity_mean, velocity_std  

# Example usage:
filename = "src/RECONSTRUCTED trajectories-400-0415_NO MOTORCYCLES.csv"
velocity_mean, velocity_std = calculate_velocity_stats(filename)
print(f"Velocity mean: {velocity_mean:.2f}, Velocity std: {velocity_std:.2f}")
