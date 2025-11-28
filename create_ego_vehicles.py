# -*- coding: utf-8 -*-
"""
Pre-processes I-80 trajectory data to identify and prepare valid merging maneuvers for simulation.

Filters out trucks and selects non-truck vehicles suitable as 'ego vehicles' based on strict criteria:

i. Starts on the on-ramp (Lane 7).
ii. Successfully merges to the outermost main lane (Lane 6).
iii. Never enters inner lanes (5, 4, 3, 2, or 1) during its trajectory.

Splits the valid ego vehicle IDs into training and testing sets (data/ego_ids).
For each ego vehicle, it generates a dedicated simulation episode CSV file
containing the vehicle's trajectory along with all environment vehicles present
on Lanes 6 and 7 during the maneuver's time window (data/vehicles).
"""
import pandas as pd
import os
from pathlib import Path
import warnings
import numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)  # Suppress future warnings

# Define paths
cwd = os.getcwd()
path = Path(cwd) / "data"
save_path = Path(path) / "vehicles"
ego_id_path = Path(path) / "ego_ids"
Path(save_path).mkdir(parents=True, exist_ok=True)
Path(ego_id_path).mkdir(parents=True, exist_ok=True)

data = pd.read_csv(Path(cwd) / "src" / "RECONSTRUCTED trajectories-400-0415_NO MOTORCYCLES.csv")
trucks = data.loc[data["Vehicle_Class_ID"] == 3].Vehicle_ID.unique()
data = data[~data["Vehicle_ID"].isin(trucks)]

# Select the vehicles that start on the ramp
vs_7 = data.loc[data['Lane_ID']==7]
vs_7_car_id = vs_7[vs_7.duplicated(subset='Vehicle_ID',keep='first')==False]
vs_7_onramp = pd.DataFrame(columns = list(data.columns))

# Filter out the vehicles that start from the onramp and
# do not merge into 5th,4th,3th,2nd and 1st lanes.
i = 0
vs_startsAt7_endsAt6 = pd.DataFrame(columns = ['Vehicle_ID'])
for v_7 in vs_7_car_id.Vehicle_ID:
    v_ = data[data['Vehicle_ID'] == v_7]
    if (v_.iloc[0].Lane_ID == 7) & (not(v_['Lane_ID'] == 5).any()) & (
            not(v_['Lane_ID'] == 4).any()) & (not(v_['Lane_ID'] == 3).any()) & (
                not(v_['Lane_ID'] == 2).any()) & (not(v_['Lane_ID'] == 1).any()):
        vs_7_onramp = vs_7_onramp._append(v_,ignore_index = True)
        vs_startsAt7_endsAt6 = vs_startsAt7_endsAt6._append({'Vehicle_ID': v_7},ignore_index = True)
        i += 1 

vs_startsAt7_endsAt6.to_csv(ego_id_path / 'vs_startsAt7_endsAt6.csv', index = None, header=True)
print("Number of cars in vs_7_onramp: " + str(i))

# Split ego IDs into training and testing sets
ego_ids = vs_startsAt7_endsAt6['Vehicle_ID']
num_test = max(1, int(0.3 * len(ego_ids)))  # Ensure at least one test sample
test_ids = np.random.choice(ego_ids, size=num_test, replace=False)
train_ids = ego_ids[~ego_ids.isin(test_ids)]

# Save the train and test IDs to CSV
pd.Series(train_ids, name="Vehicle_ID").to_csv(ego_id_path / 'train_ids.csv', index=False, header=True)
pd.Series(test_ids, name="Vehicle_ID").to_csv(ego_id_path / 'test_ids.csv', index=False, header=True)
print(f"Training set size: {len(train_ids)}")
print(f"Testing set size: {len(test_ids)}")

data = data[(data['Lane_ID'].isin([6, 7]))]

# Loop through ego vehicles and generate individual episode CSVs with corresponding environment vehicles
for ego_id in vs_startsAt7_endsAt6['Vehicle_ID']:
    ego_vehicle = data[data['Vehicle_ID'] == ego_id]
    start_frame_id = ego_vehicle['Frame_ID'].iloc[0]
    stop_frame_id = ego_vehicle['Frame_ID'].iloc[-1]  # Use the last frame ID of the ego vehicle

    # Filter environment vehicles that are present between start and stop frames of the ego vehicle
    env_vehicles_in_range = data[(data['Frame_ID'] >= start_frame_id) & (data['Frame_ID'] <= stop_frame_id)]

    # Save ego and environment vehicles to a new CSV file
    env_vehicles_in_range.to_csv(save_path / f'ego_vehicle_{int(ego_id)}_with_env.csv', index=False, header=True)
    print(f"CSV generated for ego vehicle {int(ego_id)} with environment vehicles: {len(env_vehicles_in_range)} rows.")
