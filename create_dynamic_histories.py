"""
Transforms raw vehicle history data into a structured State-Action dataset for machine learning.

Processes all generated scenario history files to extract sequential state-action pairs.
For each time step, the ego vehicle's state is calculated (relative distances/velocities
to surrounding cars in its lane and adjacent lanes, its own velocity, and distance to the merge endpoint).
The ego vehicle's observed action (Maintain, Accelerate, Decelerate, Hard Accelerate/Decelerate, or Merge)
is then inferred from its acceleration/velocity changes and lane transitions over the next time step.
All states are normalized. The resulting sequences of state-action pairs are compiled into a dictionary
and saved as a single pickle file for model training.
"""

import pandas as pd
import numpy as np
import glob
import os
import pickle
import re
from Params import Params
from datetime import datetime

seconds = 5 # duration of each scenario in seconds

def normalize_state(state):
    msg = []
    msg.append(state[0]/Params.max_sight_distance) #fs_d
    msg.append(state[1]/(Params.max_speed-Params.min_speed)) #fs_v
    msg.append(state[2]/Params.max_sight_distance) #fc_d
    msg.append(state[3]/(Params.max_speed-Params.min_speed)) #fc_v
    msg.append(state[4]/Params.max_sight_distance) #rs_d
    msg.append(state[5]/(Params.max_speed-Params.min_speed)) #rs_v
    msg.append(state[6]/(Params.max_speed-Params.min_speed)) #vel
    msg.append(float(state[7])) #lane
    if state[8]>Params.merging_region_length: #dist_end_merging
        msg.append(1)
    elif state[8]<-Params.merging_region_length: #dist_end_merging
        msg.append(-1)
    else:    
        msg.append(state[8]/Params.merging_region_length) #dist_end_merging
    return msg
def log_message(message):
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {message}")

def infer_action_from_acceleration(acceleration, ego_vel):
    """Infer the action based on the acceleration value pattern"""
    epsilon = 0.1  # Small threshold for velocity checks
    
    # Check if ego velocity is within bounds for maintain action
    if abs(acceleration) <= 0.25 and ego_vel < (Params.max_speed - epsilon) and ego_vel > (Params.min_speed + epsilon):
        return 0  # Maintain
    
    # Normal accelerate (moderate positive acceleration)
    elif 0.25 <= acceleration <= Params.actions[1][1] and ego_vel < (Params.max_speed - epsilon):
        return 1  # Accelerate
    
    # Normal decelerate (moderate negative acceleration)
    elif Params.actions[2][1] <= acceleration <= -0.25 and ego_vel > (Params.min_speed + epsilon):
        return 2  # Decelerate
    
    # Hard accelerate (high positive acceleration)
    elif acceleration >= 2.0 and ego_vel < (Params.max_speed - epsilon):
        return 3  # Hard Accelerate
    
    # Hard decelerate (high negative acceleration)
    elif acceleration <= -2.0 and ego_vel > (Params.min_speed + epsilon):
        return 4  # Hard Decelerate
    
    # If none of the above match precisely, choose the closest action
    if acceleration > 0:
        return 3 if acceleration > 1.0 else 1
    else:
        return 4 if acceleration < -1.0 else 2

def extract_scenario_ego_info(filename):
    """Extract scenario number and ego ID from filename."""
    pattern = r'history_of_scenario(\d+)_vehicle(\d+)\.csv'
    match = re.match(pattern, os.path.basename(filename))
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def get_Message(frame_id, ego_current_df, EPISODE_DF, EGO_ID):
    """Modified version of the original get_Message function that works directly with dataframes"""
    frame_df = EPISODE_DF.loc[EPISODE_DF['Frame_ID']==frame_id]
    
    ego_vel = ego_current_df['Mean_Speed'].item()
    ego_pos = ego_current_df['Local_Y'].item()
    ego_lane = ego_current_df['Lane_ID'].item()
    ego_len = ego_current_df['Vehicle_Length'].item()
    
    fs_d = Params.max_sight_distance
    fs_v = Params.max_speed - ego_vel + 0.1
    fc_d = Params.max_sight_distance
    fc_v = Params.max_speed - ego_vel + 0.1        
    rs_d = -Params.max_sight_distance
    rs_v = Params.min_speed - ego_vel - 0.1      
    car_fc = False
    car_fr = False
    
    for v_id in frame_df.Vehicle_ID:
        if v_id == EGO_ID:
            continue
        v_df = frame_df.loc[frame_df['Vehicle_ID']==v_id]
        v_vel = v_df['Mean_Speed'].item()
        v_pos = v_df['Local_Y'].item()
        v_lane = v_df['Lane_ID'].item()      
        v_len = v_df['Vehicle_Length'].item()
                
        if v_pos > Params.end_merging_point + v_len and v_lane == 0:
            continue
        
        rel_position = v_pos - ego_pos
        rel_velocity = v_vel - ego_vel
        if v_lane == (ego_lane + 1):
            if -ego_len > rel_position > rs_d:
                rs_d = rel_position + ego_len
                rs_v = rel_velocity
            elif 0 > rel_position > rs_d:
                rs_d = 0
                rs_v = rel_velocity
            elif v_len <= rel_position < fs_d:
                fs_d = rel_position - v_len
                fs_v = rel_velocity
            elif 0 <= rel_position < fs_d:
                fs_d = 0
                fs_v = rel_velocity
        elif v_lane == (ego_lane - 1):
            if -ego_len > rel_position > rs_d:
                rs_d = rel_position + ego_len
                rs_v = rel_velocity
            elif 0 > rel_position > rs_d:
                rs_d = 0
                rs_v = rel_velocity
            elif v_len <= rel_position < fs_d:
                car_fr = True
                fs_d = rel_position - v_len
                fs_v = rel_velocity
            elif 0 <= rel_position < fs_d:
                car_fr = True
                fs_d = 0
                fs_v = rel_velocity
        elif v_lane == ego_lane:
            if v_len <= rel_position < fc_d:
                car_fc = True
                fc_d = rel_position - v_len
                fc_v = rel_velocity
            elif 0 <= rel_position < fc_d:
                car_fc = True
                fc_d = 0
                fc_v = rel_velocity
            
    if (ego_lane == 0) and ((Params.end_merging_point - Params.max_sight_distance) <= ego_pos <= Params.end_merging_point) and not car_fc:
        fc_d = Params.end_merging_point - ego_pos
        fc_v = -ego_vel
    elif (ego_lane == 1) and ((Params.end_merging_point - Params.max_sight_distance) <= ego_pos <= Params.end_merging_point) and not car_fr:
        fs_d = Params.end_merging_point - ego_pos
        fs_v = -ego_vel
    if ego_lane == 1 and ego_pos >= Params.end_merging_point:
        fs_d = 0
        fs_v = 0
        
    msg = [fs_d, fs_v, fc_d, fc_v,
           rs_d, rs_v, ego_vel, 7- ego_lane,
           Params.end_merging_point - ego_pos]
  
    msg = normalize_state(msg)
    return msg




    return msg

def process_history_file(file_path):
    """Process a single history file and return the dynamic history array."""
    # Extract scenario number and ego ID from filename for logging
    scenario_no, ego_id = extract_scenario_ego_info(file_path)
    log_message(f"Processing scenario {scenario_no} with ego vehicle {ego_id}")
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    log_message(f"Loaded CSV with {len(df)} rows across {len(df['Frame_ID'].unique())} frames")
    
    # Get all frame IDs in sorted order
    frame_ids = sorted(df['Frame_ID'].unique())
    frames_to_process = frame_ids[:-1]
    
    dynamic_history = []
    total_frames = len(frames_to_process)
    
    # Process each frame except the last one
    for i, frame_id in enumerate(frames_to_process, 1):
        if i % 10 == 0:
            log_message(f"Processing frame {i}/{total_frames} for scenario {scenario_no}")
            
        # Get current frame data
        current_frame_df = df[df['Frame_ID'] == frame_id]
        ego_current_df = current_frame_df[current_frame_df['Vehicle_ID'] == ego_id]
        
        # Get next frame data
        next_frame_id = frame_ids[frame_ids.index(frame_id) + 1]
        next_frame_df = df[df['Frame_ID'] == next_frame_id]
        ego_next_df = next_frame_df[next_frame_df['Vehicle_ID'] == ego_id]
        
        # Get base message
        msg = get_Message(frame_id, ego_current_df, df, ego_id)
        
        # Check for lane change (MERGE action)
        current_lane = ego_current_df['Lane_ID'].item()
        next_lane = ego_next_df['Lane_ID'].item()
        
        if current_lane != next_lane:
            action = 5  # MERGE action
        else:
            # Calculate acceleration and infer action
            current_velocity = ego_current_df['Mean_Speed'].item()
            next_velocity = ego_next_df['Mean_Speed'].item()
            acceleration = (next_velocity - current_velocity) / 0.1  # 0.1s time step
            action = infer_action_from_acceleration(acceleration, current_velocity)
        
        # Add action to message
        msg.append(action)
        
        # Add message to dynamic history (at the beginning for newest frames first)
        dynamic_history = msg + dynamic_history
    
    log_message(f"Completed processing scenario {scenario_no} - Generated history array of length {len(dynamic_history)}")
    return scenario_no, np.array(dynamic_history, dtype=np.float32)

def create_dynamic_histories():
    """Create and save the dynamic histories dictionary."""
    log_message("Starting dynamic histories creation process")
    
    # Initialize the dictionary
    dynamic_histories = {}
    
    # Define base directory (assuming script is in the project root)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_message(f"Working directory: {base_dir}")
    
    # Construct paths
    histories_dir = os.path.join(base_dir, 'data', f'{seconds}sec_scenarios', 'histories')
    output_dir = os.path.join(base_dir, 'data', f'{seconds}sec_scenarios')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    log_message(f"Ensured output directory exists: {output_dir}")
    
    # Get all history files
    history_files = glob.glob(os.path.join(histories_dir, 'history_of_scenario*.csv'))
    total_files = len(history_files)
    log_message(f"Found {total_files} history files to process")
    
    # Process each file
    for i, file_path in enumerate(history_files, 1):
        try:
            log_message(f"Processing file {i}/{total_files}: {os.path.basename(file_path)}")
            scenario_no, dynamic_history = process_history_file(file_path)
            dynamic_histories[scenario_no] = dynamic_history
            log_message(f"Successfully added scenario {scenario_no} to dynamic histories")
        except Exception as e:
            log_message(f"ERROR processing file {file_path}: {str(e)}")
    
    # Save the dictionary
    output_path = os.path.join(output_dir, 'dynamic_histories.pickle')
    log_message(f"Saving dynamic histories to: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(dynamic_histories, f)
    
    log_message(f"Successfully saved {len(dynamic_histories)} scenarios to pickle file")
    return dynamic_histories

if __name__ == "__main__":
    log_message("Starting script execution")
    try:
        dynamic_histories = create_dynamic_histories()
        log_message("Script completed successfully")
    except Exception as e:
        log_message(f"Script failed with error: {str(e)}")