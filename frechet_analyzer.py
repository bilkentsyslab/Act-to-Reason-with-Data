"""
Before running, set your study_name and training_or_simulation variables accordingly.
This script reads Frechet distance data from a CSV file, filters out zero distances
and computes average Frechet distances for the bottom and top 50% of episodes based on the mean.
"""

import pandas as pd

study_name = "study_data_calib"
training_or_simulation = "training"  # "training" or "simulation"
csv_path = f"NGSIM_I80_Results/{study_name}/{training_or_simulation}/frechet_histories/{training_or_simulation}_frechet_distances.csv"

df = pd.read_csv(csv_path)
df = df[df["Point Distance"] != 0.0]
# Compute the overall mean
mean_distance = df["Point Distance"].mean()

# Split based on mean
top_50 = df[df["Point Distance"] <= mean_distance]
bottom_50 = df[df["Point Distance"] > mean_distance]

# Calculate group means
bottom_avg = bottom_50["Point Distance"].mean()
top_avg = top_50["Point Distance"].mean()

print(f"The frechet distances for the study: {study_name} ({training_or_simulation}):")
print("Bottom 50% (≤ mean) average:", bottom_avg)
print("Top 50% (> mean) average:", top_avg)
