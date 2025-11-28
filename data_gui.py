"""
Scenario Visualization Tool for NGSIM-Based Simulations

This script visualizes a selected simulation csv together with the real
NGSIM trajectory of the same ego vehicle.

SETUP
-----
- EPOCH_NO:      which training/simulation epoch to load
- SCENARIO_ID:   which 5-second scenario to visualize
- ITERATION_NO:  which iteration of the scenario to visualize

GHOST VEHICLE
-------------
If SHOW_GHOST = True, the real NGSIM trajectory of the ego vehicle is drawn
alongside the simulated ego. This lets you compare:
- Position difference (distance error)
- Speed difference (velocity error)

OUTPUT
------
For each frame, the viewer displays:
- Simulated ego vehicle (red)
- Real NGSIM ghost ego (white)
- Surrounding NGSIM environment vehicles (blue)
- Lane geometry of the I-80 merge section
"""

import numpy as np
import cv2
from pathlib import Path
import os
import time
import pandas as pd
import Params as Params
from src.gui.common import DEF_LANE_WIDTH, BASE_OFFSET, DRAW_SCALE, MERGE_START
from src.gui.common import CANVAS_W, CANVAS_H
from src.gui.common import fill_rectangle_scaled, write_centered
from src.gui.roads import generate_I80_merger
"""
Set the ego id, dynamic model, run no, follower status, starting road
Choose whether or not to show ghost vehicle
"""
carlength = 4.7
SHOW_GHOST = True  # Display the ego vehicle from real life data

# EGO_ID = 1064  # Vehicle ID of the ego vehicle in NGSIM data
EPOCH_NO = 0
SCENARIO_ID = 16
ITERATION_NO = 0
study_name = "study_data_original_dynamic"
filename = Path("NGSIM_I80_results") / study_name / "simulation" / "simulation_history" / "epochs" / f"epoch_{EPOCH_NO}" / f"scenario{SCENARIO_ID}-iteration{ITERATION_NO}.csv"

# FOR IDM MOBIL VISUALIZATION
# filename = Path("NGSIM_I80_results") / study_name / "simulation" / "simulation_history" / f"scenario_id_{SCENARIO_ID}" / f"{SCENARIO_ID}.csv"


# Automatically get EGO_ID from scenario file name
scenario_dir = Path("data") / "5sec_scenarios" / "scenarios"
pattern = f"scenario{SCENARIO_ID}_vehicle"
matches = [f for f in os.listdir(scenario_dir) if f.startswith(pattern)]

if len(matches) == 0:
    raise FileNotFoundError(f"No scenario file found for scenario {SCENARIO_ID}")

# Example filename: scenario6_vehicle1064.csv → extract 1064
filename_scenario = matches[0]
EGO_ID = int(filename_scenario.split("_vehicle")[1].split(".")[0])

cwd = os.getcwd()
path = os.path.join(cwd, filename)
data = pd.read_csv(path)

# Extract NGSIM data (always)
# ngsimfile = Path("data") / "NGSIM_I80_quadruplets" / FOLLOWER_STATUS / "vs_startsat7" / str(EGO_ID) / "ep.csv"
ngsimfile = Path("data") / "5sec_scenarios" / "scenarios" / f"scenario{SCENARIO_ID}_vehicle{EGO_ID}.csv"
ngsimpath = os.path.join(cwd, ngsimfile)
ngsimdata = pd.read_csv(ngsimpath)

# Extract only the ghost vehicle's data from NGSIM for ghost visualization (not used for control)
ego_ngsim_data = ngsimdata[ngsimdata["Vehicle_ID"] == EGO_ID]
ngsim_frames = np.array(ego_ngsim_data["Frame_ID"]).flatten()
ngsim_xs = np.array(ego_ngsim_data["Local_Y"]).flatten() # - carlength # * 0.3048 - 50 - carlength
ngsim_ls = np.array(ego_ngsim_data["Lane_ID"]).flatten() - 6
ngsim_vs = np.array(ego_ngsim_data["Mean_Speed"]).flatten() # * 0.3048

# Extract the environment vehicles (not the ego vehicle) from NGSIM
env_ngsim_data = ngsimdata[ngsimdata["Vehicle_ID"] != EGO_ID]
env_ids = np.array(env_ngsim_data["Vehicle_ID"]).flatten()
env_frames = np.array(env_ngsim_data["Frame_ID"]).flatten()
env_xs = np.array(env_ngsim_data["Local_Y"]).flatten() # - carlength # * 0.3048 - 50 - carlength
env_ls = np.array(env_ngsim_data["Lane_ID"]).flatten() - 6
env_vs = np.array(env_ngsim_data["Mean_Speed"]).flatten() # * 0.3048

frames = np.array(data["Frame_ID"]).flatten()
ids = np.array(data["Vehicle_ID"]).flatten()
ls = np.array(data["Lane_ID"]).flatten()
xs = np.array(data["Local_Y"]).flatten() # - carlength
vs = np.array(data["Mean_Speed"]).flatten()
accs = np.array(data["Mean_Accel"]).flatten()

# Adjust lane IDs to match the new interpretation
ls = np.where(ls == 1, 0, np.where(ls == 0, 1, ls))

class Vehicle():
    def __init__(self, x_coord, lane, velocity, width=5, length=5, is_ego=False, text_color=(200, 200, 200), color=(255, 0, 0)):
        self.x_coord = x_coord
        self.lane = lane
        self.width = width
        self.length = length
        self.velocity = velocity
        self.is_ego = is_ego
        self.text_color = text_color
        self.color = color

    def draw(self, canvas, offset=(0, 0), text="", text_color=(255, 255, 255)):
        x = self.x_coord
        y = (self.lane + 1 / 2) * DEF_LANE_WIDTH
        fill_rectangle_scaled(canvas,
                              (x - self.length / 2, y - self.width / 2),
                              (x + self.length / 2, y + self.width / 2),
                              offset=offset, color=self.color)
        write_centered(canvas, (x - self.width / 4 - 1, y + self.width / 4), text="{:.1f}".format(self.velocity), offset=offset, color=self.text_color)

class Traffic():
    def __init__(self, road, show_ghost=False):
        self.road = road
        self.show_ghost = show_ghost
        self.ghost_distance_diff = 0.0  # Initialize with default value
        self.ghost_velocity_diff = 0.0  # Initialize with default value

    def draw(self, canvas, offset=BASE_OFFSET, frame_id=0):
        self.road.draw(canvas, offset)
        write_centered(canvas, (50, 40), text=f"Frame: {frame_id}", offset=offset)

        for lane in self.road.lanes:
            for vehicle in lane.vehicles:
                vehicle.draw(canvas, offset)
        
        if self.show_ghost:
            write_centered(canvas, (50, 50), 
                        text=f"Distance Diff: {self.ghost_distance_diff:.2f} m", 
                        offset=offset)
            write_centered(canvas, (50, 60), 
                       text=f"Velocity Diff: {self.ghost_velocity_diff:.2f} m/s", 
                       offset=offset)

    def update(self, i):
        # Clear the vehicles from all lanes
        self.road.lanes[0].vehicles = []
        self.road.lanes[1].vehicles = []

        # Update the ego vehicle based on the simulation data
        ego_at = np.argwhere((frames == i) & (ids == EGO_ID))
        if len(ego_at) > 0:
            ego_x = xs[ego_at][0][0]
            ego_lane = ls[ego_at][0][0]
            ego_speed = vs[ego_at][0][0]
            self.road.lanes[ego_lane].vehicles.append(Vehicle(ego_x, ego_lane, ego_speed, is_ego=True, text_color=(0, 0, 0), color=(0, 0, 255)))

        # Add environment vehicles from NGSIM data
        env_at = np.argwhere(env_frames == i)
        for j in env_at.flatten():
            env_x = env_xs[j]
            env_lane = env_ls[j]
            env_speed = env_vs[j]
            env_vehicle_id = env_ids[j]

            # Add the environment vehicle to the lane
            if 0 <= env_lane < len(self.road.lanes):
                self.road.lanes[env_lane].vehicles.append(
                    Vehicle(env_x, env_lane, env_speed, is_ego=False, text_color=(200, 200, 200), color=(255, 0, 0))
                )

        # Draw ghost vehicle if enabled
        if self.show_ghost:
            ngsim_at = np.argwhere(ngsim_frames == i)
            if len(ngsim_at) > 0:
                ngsim_x = ngsim_xs[ngsim_at][0][0]
                ngsim_lane = ngsim_ls[ngsim_at][0][0]
                ngsim_speed = ngsim_vs[ngsim_at][0][0]

                # Check if ngsim_lane is within the valid range of lanes
                if 0 <= ngsim_lane < len(self.road.lanes):
                    ghost_vehicle = Vehicle(ngsim_x, ngsim_lane, ngsim_speed, is_ego=True, text_color=(0, 255, 0), color=(200, 200, 200))
                    self.road.lanes[ngsim_lane].vehicles.append(ghost_vehicle)
                
                # Compute the difference in distance and velocity
                self.ghost_distance_diff = abs(ngsim_x - ego_x)
                self.ghost_velocity_diff = abs(ngsim_speed - ego_speed)

def main():
    road = generate_I80_merger()
    traffic = Traffic(road, show_ghost=SHOW_GHOST)

    canvas = np.ones((CANVAS_H, CANVAS_W, 3), dtype=np.float32)
    traffic.draw(canvas)

    min_frame_id = np.min(frames)

    # if SHOW_GHOST:
    #     max_frame_id = np.max(ngsim_frames - 100)
    # else:
    #     max_frame_id = np.max(frames)

    max_frame_id = np.max(frames)

    # Continue the simulation till max_frame_id
    for i in range(min_frame_id, max_frame_id + 1):
        cv2.imshow("Traffic Simulation", canvas)
        key = cv2.waitKey(20)
        if key == ord('q'):
            break
        traffic.update(i)
        canvas[:, :, :] = 1
        traffic.draw(canvas, frame_id=i)

    mean_ego_velocity = np.mean(vs[ids == EGO_ID])  # Mean velocity for ego vehicle in simulation data
    mean_ghost_velocity = np.mean(ngsim_vs)  # Mean velocity for ego vehicle in NGSIM data

    print(f"Mean Ego Vehicle Velocity (Simulation Data): {mean_ego_velocity:.2f} m/s")
    print(f"Mean Ghost Vehicle Velocity (NGSIM Data): {mean_ghost_velocity:.2f} m/s")

    time.sleep(0.5)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
