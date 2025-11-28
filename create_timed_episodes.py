"""
Converts full vehicle episode data into standardized, sequential 5-second scenarios with 1-second histories.

Iterates through pre-generated vehicle files (each containing one ego vehicle's full trajectory and surrounding traffic)
and slices them into fixed-length (50-frame) scenarios. It filters out any segments containing sudden vehicle appearances
in the critical Lane 6 area to ensure data quality. The script saves the scenario (data/5sec_scenarios/scenarios) and
its preceding history (data/5sec_scenarios) as separate files, normalizes coordinates (feet to meters, applies offset),
and finally generates a train/test split for the numbered scenarios.
"""
import pandas as pd
import os
from pathlib import Path
import warnings
from typing import List, Dict, Tuple
import random
from Params import Params
import shutil

warnings.simplefilter(action='ignore', category=FutureWarning)

seconds = 5 # duration of each scenario in seconds
frames_per_second = 10
total_frames = frames_per_second * seconds  # 50 frames for 5 seconds

def setup_directories() -> Tuple[Path, Path, Path]:
    """Setup necessary directories for data processing."""
    cwd = os.getcwd()
    base_path = Path(cwd) / "data"
    vehicles_path = base_path / "vehicles"
    save_path = base_path / f"{seconds}sec_scenarios"
    scenario_save_path = save_path / "scenarios"
    history_save_path = save_path / "histories"
    
    # Clear directories if they exist
    for path in [scenario_save_path, history_save_path]:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    
    return vehicles_path, scenario_save_path, history_save_path

def check_sudden_appearances(data: pd.DataFrame, start_frame: int, end_frame: int, 
                           entrance_threshold: float = 15) -> bool:
    """
    Check if there are any sudden appearances in Lane 6 between start_frame and end_frame.
    Vehicles merging from Lane 7 to Lane 6 are not counted as sudden appearances.
    """
    frames = sorted(data[(data['Frame_ID'] >= start_frame) & 
                        (data['Frame_ID'] <= end_frame)]['Frame_ID'].unique())
    
    for i in range(1, len(frames)):
        prev_frame = frames[i-1]
        curr_frame = frames[i]
        
        prev_frame_data = data[data['Frame_ID'] == prev_frame]
        curr_frame_data = data[data['Frame_ID'] == curr_frame]
        
        # Get vehicles in each frame
        prev_vehicles = set(prev_frame_data['Vehicle_ID'])
        curr_vehicles = set(curr_frame_data['Vehicle_ID'])
        
        # Check new vehicles
        new_vehicles = curr_vehicles - prev_vehicles
        for vehicle in new_vehicles:
            vehicle_data = curr_frame_data[curr_frame_data['Vehicle_ID'] == vehicle]
            # Only consider it a sudden appearance if it appears in Lane 6 beyond threshold
            if (vehicle_data['Lane_ID'].iloc[0] == 6 and 
                vehicle_data['Local_Y'].iloc[0] > entrance_threshold):
                return False
    
    return True


def create_scenarios_from_file(file_path: Path, ego_id: int, scenario_frames: int = total_frames,
                             history_frames: int = 10) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create 5-second scenarios with 1-second histories from a vehicle file.
    Last frame of history is the first frame of scenario.
    """
    data = pd.read_csv(file_path)
    
    
    # Convert values from feet to meter.
    data["Mean_Speed"] = data["Mean_Speed"] * 0.3048
    data["Mean_Accel"] = data["Mean_Accel"] * 0.3048
    data["Vehicle_Length"] = data["Vehicle_Length"] * 0.3048
    
    # Besides converting the value from feet the meter, subtract the offset to make beginning Local_Y 0.
    data["Local_Y"] = ( data["Local_Y"] * 0.3048 ) - Params.offset
    
    data = data[data['Lane_ID'].isin([6, 7])]
    
    all_frames = sorted(data['Frame_ID'].unique())
    valid_scenarios = []
    
    current_frame_idx = history_frames - 1
    while current_frame_idx < len(all_frames) - scenario_frames:
        start_frame = all_frames[current_frame_idx]
        end_frame = all_frames[current_frame_idx + scenario_frames - 1]
        history_start_frame = all_frames[current_frame_idx - (history_frames - 1)] 
        
        scenario_data = data[
            (data['Frame_ID'] >= start_frame) &
            (data['Frame_ID'] <= end_frame)
        ].copy()
        
        history_data = data[
            (data['Frame_ID'] >= history_start_frame) &
            (data['Frame_ID'] <= start_frame)
        ].copy()
        
        history_valid = check_sudden_appearances(
                data, history_start_frame, start_frame
            )
        scenario_valid = check_sudden_appearances(
                data, start_frame, end_frame
            )
        
        # Ensure ego vehicle's last Local_Y <= 315
        ego_scenario_data = scenario_data[scenario_data['Vehicle_ID'] == ego_id]
        ego_last_local_y = ego_scenario_data['Local_Y'].iloc[-1]
        ego_within_limit = ego_last_local_y <= Params.end_for_car0 - 50 #TODO might reject the last ones too much or else create error
        
        if history_valid and scenario_valid and ego_within_limit:
            valid_scenarios.append((scenario_data, history_data))
            current_frame_idx += scenario_frames
        else:
            current_frame_idx += 1
    
    return valid_scenarios

def save_scenarios_and_histories(scenarios: List[Tuple[pd.DataFrame, pd.DataFrame]], 
                               ego_id: str,
                               start_scenario_num: int,
                               scenario_path: Path,
                               history_path: Path) -> int:
    """
    Save valid scenarios and histories to their respective directories.
    Returns the next available scenario number.
    """
    current_scenario_num = start_scenario_num
    
    for scenario_data, history_data in scenarios:
        # Save scenario
        scenario_filename = f"scenario{current_scenario_num}_vehicle{ego_id}.csv"
        scenario_data.to_csv(scenario_path / scenario_filename, index=False)
        
        # Save history
        history_filename = f"history_of_scenario{current_scenario_num}_vehicle{ego_id}.csv"
        history_data.to_csv(history_path / history_filename, index=False)
        
        current_scenario_num += 1
    
    return current_scenario_num

def create_train_test_split(total_scenarios: int, train_ratio: float = 0.8) -> None:
    """
    Create train and test splits from the scenario numbers.
    """
    scenario_numbers = list(range(total_scenarios))
    random.shuffle(scenario_numbers)
    
    split_idx = int(total_scenarios * train_ratio)
    train_scenarios = scenario_numbers[:split_idx]
    test_scenarios = scenario_numbers[split_idx:]
    
    # Save train scenarios
    pd.DataFrame(train_scenarios, columns=["Scenarios"]).sort_values(by="Scenarios").to_csv(
        Path(f"data/{seconds}sec_scenarios/train_scenarios.csv"), index=False)
    
    # Save test scenarios
    pd.DataFrame(test_scenarios, columns=["Scenarios"]).sort_values(by="Scenarios").to_csv(
        Path(f"data/{seconds}sec_scenarios/test_scenarios.csv"), index=False)

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Setup directories
    vehicles_path, scenario_save_path, history_save_path = setup_directories()
    
    # Process each vehicle file
    current_scenario_num = 0
    for vehicle_file in vehicles_path.glob("ego_vehicle_*_with_env.csv"):
        # Extract ego_id from filename
        ego_id = vehicle_file.stem.split("_")[2]
        
        # Create scenarios from this vehicle's data
        valid_scenarios = create_scenarios_from_file(vehicle_file, ego_id = int(ego_id))
        
        # Save scenarios and histories, get next available scenario number
        current_scenario_num = save_scenarios_and_histories(
            valid_scenarios, ego_id, current_scenario_num,
            scenario_save_path, history_save_path
        )
        
        print(f"Processed vehicle {ego_id}: Created {len(valid_scenarios)} scenarios")
    
    # Create train/test split
    create_train_test_split(current_scenario_num)
    print(f"Created train/test split for {current_scenario_num} total scenarios")

if __name__ == "__main__":
    main()
