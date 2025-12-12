"""
Steps:
Set MODEL_LEVEL1, MODEL_LEVEL2, MODEL_LEVEL3, MODEL_DYNAMIC model numbers. These are related to level-k models.
Do Training.
Before doing simulation, choose MODEL_DATA and EPOCH_NO depending on where the latest model of the data training belongs to.
Set Simulate = True, Train = False
"""
from Params import Params
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import warnings  
import shutil
import pickle
warnings.simplefilter(action='ignore', category=FutureWarning) # not to see futurewarning warning

Simulate = False
Train = True

if Simulate:
    from src.data_DynamicDQNAgent_sim import DynamicDQNAgent
if Train:
    from src.data_DynamicDQNAgent import DynamicDQNAgent

MODEL_DATA = 10824 # The model from data that will be used # for simulation
EPOCH_NO = "epoch_4" # which env no the model above belongs to # for simulation
NUM_EPOCHS = 5 if Train else 1
SCENARIO_ITERATION = 5
NUM_SCENARIOS = 433 #ONLY FOR TRAINING
boltzmann_decay_end_state = NUM_EPOCHS*SCENARIO_ITERATION*NUM_SCENARIOS // 2 # multiply it by episode numbers per num epochs 

MODEL_LEVEL1 = 99 # Model number of the level1 DQN
MODEL_LEVEL2 = 99 # Model number of the level2 DQN
MODEL_LEVEL3 = 99 # Model number of the level3 DQN
MODEL_DYNAMIC = 99  # Model number of the dynamic DQN

SKIP_FRAMES = int(Params.timestep * 10)

BASE_DIR = Path(__file__).parent.resolve() # Root path for your ATR Data
DATA_PATH = BASE_DIR / "data"

TIMED_DATA_PATH = DATA_PATH / "5sec_scenarios"
TIMED_SCENARIOS_PATH = TIMED_DATA_PATH / "scenarios"
SCENARIO_HISTORIES_PATH = TIMED_DATA_PATH / "histories"
EGO_DYNAMIC_HISTORIES_PATH = TIMED_DATA_PATH / "dynamic_histories.pickle"

NGSIM_RESULTS_PATH = BASE_DIR / "NGSIM_I80_Results"
ATR_ORIGINAL_PATH = BASE_DIR / "ATR_Original" 
EXPERIMENTS_PATH = ATR_ORIGINAL_PATH / "experiments"

STUDY_TITLE = "study_data_calib" # you can change this.
STUDY_PATH = NGSIM_RESULTS_PATH / STUDY_TITLE
DATA_TRAINING_PATH = STUDY_PATH / "training"
DATA_SIMULATION_PATH = STUDY_PATH / "simulation"

AGENT_DATA_PATH = STUDY_PATH / "training" / "agent_data"
MODELS_DIR = STUDY_PATH / "training" / "models"
TARGET_WEIGHTS_PATH = STUDY_PATH / "training" / "target_weights"
TRAINING_HISTORY_PATH = STUDY_PATH / "training" / "training_history"
SIMULATION_HISTORY_PATH = STUDY_PATH / "simulation" / "simulation_history"

FRECHET_TRAINING_PATH = Path(DATA_TRAINING_PATH) / "frechet_histories"
FRECHET_TRAINING_PATH.mkdir(parents=True, exist_ok=True)
FRECHET_SIMULATION_PATH = Path(DATA_SIMULATION_PATH) / "frechet_histories"
FRECHET_SIMULATION_PATH.mkdir(parents=True, exist_ok=True)

ENABLE_BOLTZMANN_SAMPLING = True 
ENABLE_DYNAMIC_DRIVING_BOLTZMANN_SAMPLING = True
ENABLE_RANDOM_DYNAMIC_STRATEGY = False # the code is set up for always False case

EGO_DF_DYN_COLS = ['Scenario_ID','Episode', 'Time_Step', 'State', 'fs_d', 'fs_v', 'fc_d', 
                   'fc_v', 'rs_d', 'rs_v', 'velocity', 'lane', 'dist_end_merging', 
                   'q_value1_acc1', 'q_value1_acc_0', 'q_value1_acc_minus1',
                   'q_value2_acc1', 'q_value2_acc_0', 'q_value2_acc_minus1',
                   'q_value3_acc1', 'q_value3_acc_0', 'q_value3_acc_minus1',
                    'Merge',
                    'q-maintain1', 'q-accel1', 
                   'q-decel1', 'q-hard_accel1', 'q-hard_decel1', 'q-merge1',
                   'q-maintain2', 'q-accel2', 'q-decel2', 'q-hard_accel2', 
                   'q-hard_decel2', 'q-merge2', 'q-maintain3', 'q-accel3', 
                   'q-decel3', 'q-hard_accel3', 'q-hard_decel3', 'q-merge3', 
                   'Dynamic_Action', 'Dynamic_Action_Type', 'Action']
EGO_LOG_PATH = TRAINING_HISTORY_PATH / "ego_log.csv" if Train else SIMULATION_HISTORY_PATH / "ego_log.csv"

FINAL_STATE_DF_COLUMNS_TRAINING = ['Run_No','Final_State','Real_Final_Step','Training_Final_Step']
FINAL_STATE_DF_COLUMNS_SIMULATION = ['Run_No','Final_State','Real_Final_Step','Simulation_Final_Step']

STATE_SIZE = Params.num_observations
ACTION_SIZE = Params.num_actions
DYNAMIC_STATE_SIZE = Params.num_observations * Params.dynamic_history + \
    Params.dynamic_history_action * (Params.dynamic_history-1)
DYNAMIC_ACTION_SIZE = Params.num_dynamic_actions

DYNAMIC_AGENT = None
EPISODE_DF = None
EGO_ID = None

# For the 5 second scenario implementation
DYNAMIC_HISTORIES = None

# EGO_IDs = pd.read_csv(DATA_PATH / "ego_ids" / "vs_always6_final_filtered.csv")
# EGO_IDs = pd.concat([EGO_IDs, pd.read_csv(DATA_PATH / "ego_ids" / "vs_startsAt7_endsAt6_final_filtered.csv")], axis=0)

collision_episode_keeper = []
q_loss_table = [] 
reward_table = []
point_distances = []

def frechet(p, q):
    n_p, n_q = len(p), len(q)
    dp = np.full((n_p, n_q), -1.0)

    def compute(i, j):
        if dp[i, j] > -1:
            return dp[i, j]
        if i == 0 and j == 0:
            dp[i, j] = abs(p[0] - q[0])
        elif i == 0:
            dp[i, j] = max(compute(i, j - 1), abs(p[i] - q[j]))
        elif j == 0:
            dp[i, j] = max(compute(i - 1, j), abs(p[i] - q[j]))
        else:
            dp[i, j] = max(
                min(compute(i - 1, j), compute(i - 1, j - 1), compute(i, j - 1)),
                abs(p[i] - q[j]),
            )
        return dp[i, j]

    return compute(n_p - 1, n_q - 1)

def moving_average(data, window_size):
    """Compute the moving average with a specified window size."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

#Normalize state variables
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

def remember_frame(currentstate, actionget,  state_messages, reward, collided, 
                       ego_reached_end, state_size, dynamic_actionget):
        """
        Appends a given transition to the memory

        Parameters
        ----------
        currentstate : numpy array
            Current state of the ego.
        actionget : int
            Current driving action of the ego.
        dynamic_actionget : int
            Current level-k action of the dynamic ego.
        state_messages : numpy array
            Next state of the ego.
        reward : float
            Reward taken with the current action.
        collided : bool
            True if ego is in a collision.
        ego_reached_end : bool
            True if ego reached the end.
        state_size : int
            Size of the state, i.e. the input layer.

        Returns
        -------
        None.

        """
        # Dynamic ego
        state_size = 9
        temp_state_messages = state_messages.copy()
        state_messages = currentstate.copy()
        if Params.dynamic_history_action:
            scale_action = 1.0 if not Params.scale_action else (Params.num_actions-1.0)
            state_messages_concat = np.concatenate((state_messages[0,0,:state_size].copy(),
                                                [actionget/scale_action],
                                                state_messages[0,0,state_size:-10].copy()))
            state_messages[0,0,state_size:] = state_messages_concat
        else:
            state_messages[0,0,state_size:] = state_messages[0,0,:-state_size].copy()
        state_messages[:,:, :state_size] = temp_state_messages.copy()
        DYNAMIC_AGENT.remember(currentstate, 
                                        dynamic_actionget, 
                                        reward, state_messages, 
                                        (collided or ego_reached_end))

def get_reward(crash, state_messages, ego_velocity, ego_lane, fc_d, dist_end_merging, x_training, x_data, lane_training, lane_data, acc, merged, real_trajectory, simulated_trajectory, is_terminal):
    performance = 1.0
    scale = 0.2 # 0.02
    
    wc = 1000 * scale # 1000 # Collision
    wv = 10 * scale * performance # Velocity
    we = 5 * scale #*performance # Effort
    wh = 5 * scale # Headway
    wnm = 5 * scale #*performance # Not Merging
    ws = 30 * scale # 100  #  *performance # Velocity Less than 2.25m/s or Stopping on Lane-0 with dist_end_merging less than far distance
    w7 = 200 * scale #TODO tune
    w8 = -50 * scale #TODO tune

    # Collision parameter
    c = 0
    if crash:
        c = -1
    
    # Velocity parameter
    ego_velocity *= (Params.max_speed-Params.min_speed)
    v_coeff = 1 # 0.2
    if ego_velocity > Params.nominal_speed:
        #v = v_coeff * (Params.max_speed - ego_velocity) / (Params.max_speed - Params.nominal_speed)
        v = v_coeff * (ego_velocity - Params.nominal_speed) / (Params.max_speed - Params.nominal_speed)
        # v = 1 at nominal speed, 0 at max speed
    else:
        # Punishment for below nominal speed
        v = v_coeff * (ego_velocity - Params.nominal_speed) / (Params.nominal_speed - Params.min_speed)
        # v = 0 at nominal speed, -1 at min speed

    # Effort parameter
    if ego_velocity > (Params.nominal_speed+Params.min_speed)/2:
        if acc >= 2.47 or acc <= -3.591:
            e = -1
        elif acc >= 1.25 or acc <= -1.1:
            e = -0.25
        else:
            e = 0
    else:
        e = 0
    
    # Headway parameter
    h = 0
    fc_d = state_messages[2]*Params.max_sight_distance # denormalize
    # Case 1: If the front car distance is less than or equal to the close distance (d_close)
    if fc_d <= Params.close_distance:
        h = -1  # Penalize heavily if the ego vehicle is too close to the front car
    # Case 2: If the front car distance is between the close and nominal distances
    elif fc_d <= Params.nominal_distance:
        h = (fc_d - Params.nominal_distance) / (Params.nominal_distance - Params.close_distance)
        # The reward should decrease as the ego vehicle gets closer to d_close
    # if the ego velocity is not very small, give reward for the big headway distance
    elif ego_velocity > -Params.hard_decel_rate*Params.timestep:
        # Case 3: If the front car distance is between the nominal and far distances
        if fc_d <= Params.far_distance:
            h = (fc_d - Params.nominal_distance) / (Params.far_distance - Params.nominal_distance)
        # The reward increases as ego gets relatively far from the front car
        # Case 4: If the front car distance is greater than the far distance
        elif fc_d > Params.far_distance:
            h = 1

    # Not Merging parameter
    nm = 0
    if ego_lane == 0: # and 0 < dist_end_merging < 1:  # lets keep it the og way
        nm = -1 

    # Stopping parameter
    fl_d = state_messages[0]*Params.max_sight_distance # denormalizing
    rl_d = state_messages[4]*Params.max_sight_distance # denormalizing
    dist_end_merging = state_messages[-1]*Params.merging_region_length  # denormalizing

    s = 0
    # Main Road Case
    if ego_velocity < -Params.hard_decel_rate*Params.timestep: 
        if ego_lane == 1:  # Ego is on the main road
            # Penalize if the ego is too far behind, not accelerating, or decelerating unnecessarily
            if acc <= 1.25 and fc_d >= Params.far_distance and (dist_end_merging >= Params.far_distance or dist_end_merging <= 0):
                s = -1 # Penalize for not accelerating in safe conditions
            else:
                s = 0  # No penalty if conditions are met
                # print("Main road: No penalty")

        # Ramp Case
        elif ego_lane == 0:  # Ego is on the ramp
            # Inside the merging region
            if dist_end_merging < Params.merging_region_length:
                # Penalize if not merging while it's safe
                if merged != True:
                    if fl_d >= Params.close_distance and abs(rl_d) >= (1.5 * Params.far_distance):
                        s = -1  # Penalize for not merging when safe
                        # print("Ramp: Penalize for not merging in safe conditions")
                    elif dist_end_merging <= Params.far_distance:
                        s = -0.05  # Small penalty for stopping too close to the merging zone end
                        # print("Ramp: Small penalty for being close to merging zone end")
                else:
                    s = 0  # No penalty for merging

            # Outside the merging region
            else:
                if fc_d >= Params.far_distance:
                    s = -1  # Penalize for not accelerating or preparing to merge

    ox = 0

    # lambda_x = 0.05
    # if len(simulated_trajectory) >= 2 or len(real_trajectory) >= 2:
    #     frechet_prev = abs(simulated_trajectory[-2] - real_trajectory[-2])
    #     frechet_now = abs(simulated_trajectory[-1] - real_trajectory[-1])

    #     ox = (frechet_prev-frechet_now) - lambda_x * frechet_now

    # F_MAX = 20.0
    # if len(simulated_trajectory) >= 2 or len(real_trajectory) >= 2:
    #     frechet_now = abs(simulated_trajectory[-1] - real_trajectory[-1])
    #     d = min(frechet_now, F_MAX)
    #     ox = -(d / F_MAX) # normalize between -1 and 0

    F_MAX = 30.0
    if is_terminal and len(simulated_trajectory) >= 2:
        frechet_dist = frechet(real_trajectory, simulated_trajectory)
        d = min(frechet_dist, F_MAX)
        ox = -(d / F_MAX)

    # Calibration term for y-axis (lane offset oy)
    oy = 1 if int(lane_training) != lane_data else 0

    # Final reward with calibration terms
    R = wc * c + wv * v + we * e + wh * h + wnm * nm + ws * s
    R_calib = R + ox * w7 + oy * w8 

    return R_calib

def update_motion(ego_df, frame_id, act, dynamic_actionget):
    
    ego_class_id = ego_df['Vehicle_Class_ID'].iloc[-1]
    ego_len = ego_df['Vehicle_Length'].iloc[-1]
    ego_follower_id = ego_df['Follower_ID'].iloc[-1]
    ego_leader_id = ego_df['Leader_ID'].iloc[-1]
    ego_pos = ego_df['Local_Y'].iloc[-1]    
    ego_lane = ego_df['Lane_ID'].iloc[-1]
    ego_vel = ego_df['Mean_Speed'].iloc[-1]
        
    acc = 0 
    epsilon = 0.1
    timestep = Params.timestep/SKIP_FRAMES
    merged = False
    
    new_lane = ego_lane
    new_vel = ego_vel
    new_pos = ego_pos
    extra_acc = (1)*( 1 - (dynamic_actionget) % 3)*(act!= 5)

    # Merge
    if act == 5:
        new_lane += 1
        merged = True
    #Implements accelerate action by sampling an acceleration
    elif act == 1 and ego_vel < (Params.max_speed - epsilon):
        acc = min(0.25 + np.random.exponential(scale=0.75), Params.actions[act][1])

    #Implements decelerate action by sampling an acceleration
    elif act == 2 and ego_vel> (Params.min_speed + epsilon):
        acc = max(-0.25 - np.random.exponential(scale=0.75),Params.actions[act][1])

    #Implements hard accelerate action by sampling an acceleration
    elif act == 3 and ego_vel< (Params.max_speed - epsilon):
        acc = min(2 + np.random.exponential(scale=0.75),Params.actions[act][1])

    #Implements hard decelerate action by sampling an acceleration
    elif act == 4 and ego_vel > (Params.min_speed + epsilon):
        acc = max(-2 - np.random.exponential(scale=0.75),Params.actions[act][1])

    #Implements maintain action by sampling an acceleration value
    elif act == 0 and (ego_vel < (Params.max_speed - epsilon)) and (
        ego_vel > (Params.min_speed + epsilon)):
        acc = np.random.laplace(scale=0.1)
        acc = (acc>=0)*min(acc,0.25)+(acc<0)*max(acc,-0.25)

    acc += extra_acc
    for idx, new_frame in enumerate(range(frame_id+1, frame_id+SKIP_FRAMES+1)):
        new_vel += acc * timestep
        #Assign the velocity within limits
        if new_vel < Params.min_speed:
            new_vel = Params.min_speed
        elif new_vel > Params.max_speed:
            new_vel = Params.max_speed
    
        # Update current position TODO
        if new_vel == Params.min_speed:
            new_pos += (ego_vel + 0.5*(Params.min_speed-ego_vel))*timestep
        elif new_vel == Params.max_speed:
            new_pos+= (ego_vel + 0.5*(Params.max_speed-ego_vel))*timestep
        else:
            new_pos += (ego_vel+0.5*(acc*timestep))*timestep
            
        temp_lane = ego_lane
        if idx == SKIP_FRAMES - 1:            
            temp_lane = new_lane

        ego_df = ego_df._append({"Vehicle_ID": EGO_ID,
                                "Frame_ID": new_frame,
                                "Lane_ID": temp_lane,
                                "Local_Y": new_pos,
                                "Mean_Speed": new_vel,
                                "Mean_Accel": acc,
                                "Vehicle_Length": ego_len,
                                "Vehicle_Class_ID": ego_class_id, 
                                "Follower_ID": ego_follower_id,
                                "Leader_ID": ego_leader_id}, ignore_index=True);
        ego_vel = new_vel
        
    # Check if the ego has passed the end of the road
    reached_end = new_pos > Params.end_for_car0        
    #print(f"Frame: {new_frame}, Position: {new_pos}, Velocity: {new_vel}, Acceleration: {acc}")

    return reached_end, ego_df , acc, merged

def select_path(is_simulation=False):
    global STUDY_RES_DIR

    STUDY_RES_DIR = SIMULATION_HISTORY_PATH if is_simulation else TRAINING_HISTORY_PATH

    STUDY_RES_DIR.mkdir(parents=True, exist_ok=True)

def get_Message(frame_id,ego_current_df,ignored_rear_vehicles, normalize=True):
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
        if v_id in ignored_rear_vehicles:
            continue  # Skip this vehicle if it's in the ignored rear vehicles set

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
           rs_d, rs_v, ego_vel, ego_lane,
           Params.end_merging_point - ego_pos]
    # print(frame_id, fc_d)
    if normalize:
        msg = normalize_state(msg)
    return msg


def collision_check(frame_id,ego_df, ignored_rear_vehicles):

    ego_len = ego_df['Vehicle_Length'].iloc[-1]

    for temp_frame_id in range(frame_id+1,frame_id+SKIP_FRAMES+1):
        ego_pos = ego_df.loc[ego_df['Frame_ID']==temp_frame_id].Local_Y.item()
        ego_lane = ego_df.loc[ego_df['Frame_ID']==temp_frame_id].Lane_ID.item()        
        
        frame_df = EPISODE_DF.loc[EPISODE_DF['Frame_ID']==temp_frame_id]
        for v_id in frame_df.Vehicle_ID:   #all the vehicles in that frame 
            if v_id == EGO_ID:
                if ego_lane == 0 and (ego_pos >= Params.end_merging_point):
                    return [True, 0, None] # "Ego Didn't Merge"
                continue    
            
            # Skip processing for any vehicle in ignored_rear_vehicles
            if v_id in ignored_rear_vehicles:
                continue  # Skip this vehicle and go to the next one in the loop

            v_df = EPISODE_DF.loc[EPISODE_DF['Vehicle_ID']==v_id]
            v_len = v_df.loc[v_df['Frame_ID']==temp_frame_id].Vehicle_Length.item()
            v_pos = v_df.loc[v_df['Frame_ID']==temp_frame_id].Local_Y.item()
            v_lane = v_df.loc[v_df['Frame_ID']==temp_frame_id].Lane_ID.item()
            # CURRENT AND PREVIOUS ACTIONS ?
            if (-ego_len <= v_pos-ego_pos <= (v_len + 0.1) and ego_lane == v_lane):
                for temp_past_frame_id in range(frame_id+1,temp_frame_id):
                    if (v_df['Frame_ID']==temp_past_frame_id).any():
                        v_lane_past = v_df.loc[v_df['Frame_ID']==temp_past_frame_id].Lane_ID.item()
                        if (v_lane_past != v_lane):
                            return [True, 1, None] # A car merged into ego, check if the vehicle changed lanes
                    ego_lane_past = ego_df.loc[ego_df['Frame_ID']==temp_past_frame_id].Lane_ID.item()
                    if (ego_lane != ego_lane_past):
                        return [True, 2, None] # Ego merged into a car 
                if v_pos > ego_pos:
                    return [True, 3, None] # Ego Crashed
                elif ego_pos > v_pos:
                    return [True, 5, v_id] # Rear-Ended / environment vehicle crashed into ego
                else:
                    print("Unindentified Collision at step "+str(temp_frame_id))
    return [False, -1, None]        

def load_dynamic_model(is_simulation=False):
    global DYNAMIC_AGENT

    MODEL_CONFIG = {"models": {1: MODEL_LEVEL1,
                               2: MODEL_LEVEL2,
                               3: MODEL_LEVEL3,
                               4: MODEL_DYNAMIC}}
    
    agent_levelk_paths = {1: None, 2: None, 3: None}

    for i in range(1, 4):
        agent_levelk_paths[i] = Path(EXPERIMENTS_PATH) / "some_title" / f"level{i}" / \
                                "training" / "models" / f"model{MODEL_CONFIG['models'][i]}"

    levelk_config = {
        "paths": agent_levelk_paths,
        "boltzmann_sampling": ENABLE_DYNAMIC_DRIVING_BOLTZMANN_SAMPLING
    }

    DYNAMIC_AGENT = DynamicDQNAgent(DYNAMIC_STATE_SIZE,
                                    DYNAMIC_ACTION_SIZE,
                                    STATE_SIZE,
                                    ACTION_SIZE,
                                    levelk_config)
    
    if not ENABLE_RANDOM_DYNAMIC_STRATEGY:
        model_path = Path(EXPERIMENTS_PATH) / "some_title" / "dynamic" / "training" / "models" / f"model{MODEL_DYNAMIC}" # This model is from level-k
        if is_simulation:
            model_path = MODELS_DIR / EPOCH_NO / f"model{MODEL_DATA}.h5" # This model is from data training
        
        DYNAMIC_AGENT.load(model_path)

    if is_simulation:
        DYNAMIC_AGENT.T = int(1) # it was min t
    else:
        DYNAMIC_AGENT.T = int(65)
    DYNAMIC_AGENT.boltzmann_sampling = ENABLE_BOLTZMANN_SAMPLING

def load_episode_df(scenario_id):
    global EGO_ID
    global EPISODE_DF
    
    # Pull the filename that matches the name of the scenario id and then...
    # extract the ego id from that filename    
    def find_scenario_file_and_ego_id(scenario_id, directory):
        # Pattern to match: scenario{scenario_id}_vehicle*.csv
        pattern = f"scenario{scenario_id}_vehicle*.csv"
        
        # Search for matching files
        matching_files = list(Path(directory).glob(pattern))
        
        if not matching_files:
            raise FileNotFoundError(f"No file found for scenario {scenario_id}")
        
        if len(matching_files) > 1:
            raise ValueError(f"Multiple files found for scenario {scenario_id}")
        
        # Get the filename as string
        filename = str(matching_files[0].name)
        
        # Extract the vehicle ID
        # Split by 'vehicle' and take the second part, then remove '.csv'
        vehicle_id = int(filename.split('vehicle')[1].replace('.csv', ''))
        
        return filename, vehicle_id
    
    scenario_filename, ego_id = find_scenario_file_and_ego_id(scenario_id, TIMED_SCENARIOS_PATH)
    
    EGO_ID = ego_id

    scenario_path = TIMED_SCENARIOS_PATH / scenario_filename
    EPISODE_DF = pd.read_csv(scenario_path)
    
    # data / NGSIM_I80_quadruplets / no_followers / vs_always6 / 2171 / ep_follower_2178.csv'
    EPISODE_DF.Lane_ID = 7-EPISODE_DF.Lane_ID

def run_episode(ego_log, total_timesteps, final_state_df, episode_no, run_no, ego_count, epoch_no, scenario_id, o, save_dir=""):
    global DYNAMIC_HISTORIES

    # Initialize a set to keep track of ignored rear vehicles
    ignored_rear_vehicles = set()
    
    # Initialize properly the ego dynamic history
    ego_dyn_state = np.concatenate( [
        np.zeros((1, 1, 9)),
        DYNAMIC_HISTORIES[scenario_id].reshape(1, 1, -1) ], axis=2 ) # np.zeros((1, 1, DYNAMIC_STATE_SIZE))
    DYNAMIC_HISTORIES = DYNAMIC_HISTORIES
    hist_action_size = int(Params.dynamic_history_action)
    counter = 0
    state_size = 10 
    episode_no = episode_no
    total_timesteps = total_timesteps
    actionget = 0
    new_episode_df = EPISODE_DF.copy() # these are already converted to SI

    # Get the initial state of the ego vehicle
    ego_df = EPISODE_DF.loc[EPISODE_DF['Vehicle_ID'] == EGO_ID].iloc[[0]].copy() # extracts the first row of the ego vehicle from the data csv # double brackets are for keeping it as a dataframe
    ego_len = ego_df['Vehicle_Length'].item()
    
    frames = pd.Series(EPISODE_DF.loc[EPISODE_DF['Vehicle_ID'] == EGO_ID, 'Frame_ID'].unique()) # frame ids of ego vehicle
    frames_y = EPISODE_DF.loc[EPISODE_DF['Vehicle_ID'] == EGO_ID, 'Local_Y'] # local ys of ego vehicle

    simulated_trajectory = []
    real_trajectory = []

    first_execution = True
    prev_frame_id = frames.iloc[0]
    for frame_id in frames:
        
        first_collision = True
        
        # Skip frames that don't meet the SKIP_FRAMES criteria
        if not (frame_id == frames.iloc[0]) and 0 < frame_id - prev_frame_id < SKIP_FRAMES:
            continue

        # Initial ego state
        if frame_id == frames.iloc[0]:
            ego_position = ego_df['Local_Y'].item()
            ego_lane = ego_df['Lane_ID'].item()
            
            currentstate_list = get_Message(frame_id=frame_id, ego_current_df=ego_df, ignored_rear_vehicles=ignored_rear_vehicles, normalize=True)  # Get current state
        else:
            ego_position = ego_df.iloc[[-1]]['Local_Y'].item()
            ego_lane = ego_df.iloc[[-1]]['Lane_ID'].item()
            currentstate_list = get_Message(frame_id=frame_id, ego_current_df=ego_df.iloc[[-1]], ignored_rear_vehicles=ignored_rear_vehicles, normalize=True)

        currentstate = np.reshape(currentstate_list, [1, 1, STATE_SIZE])

        # Check if merging is required based on ego position and lane
        # remove_merging parameter determines whether to remove the "merge" action from the last layer
        if (Params.start_merging_point + ego_len) < ego_position < Params.end_merging_point and ego_lane == 0:
            remove_merging = False
        else:
            remove_merging = True

        if Params.dynamic_history_action:
            scale_action = 1.0 if not Params.scale_action else (Params.num_actions - 1.0)

            if not first_execution:
                ego_dyn_state[0, 0, STATE_SIZE:] = np.concatenate((
                ego_dyn_state[0, 0, :STATE_SIZE:].copy(), 
                [actionget / scale_action],
                ego_dyn_state[0, 0, STATE_SIZE:-state_size].copy()
            ))
            
            if first_execution:
                # for s_i in range(Params.dynamic_history, 1, -1):
                #     idx_1 = ((s_i - 1) * state_size - hist_action_size)
                #     idx_2 = s_i * state_size - 2 * hist_action_size
                #     ego_dyn_state[0, 0, idx_1:idx_2] = currentstate[0, 0, :].copy()
                first_execution = False

        else:
            ego_dyn_state[0, 0, STATE_SIZE:] = ego_dyn_state[0, 0, :-STATE_SIZE:].copy()

        # Always update the first STATE_SIZE values with current state
        ego_dyn_state[0, 0, 0:STATE_SIZE] = currentstate[0, 0, :].copy()

        # Update currentstate with the dynamic state
        currentstate = ego_dyn_state.copy()

        # Get the action for the ego vehicle from the dynamic agent
        ego_info = DYNAMIC_AGENT.act(state=currentstate, remove_merging=remove_merging, get_qvals=True)

        # Extract action information
        dynamic_action_type = ego_info[0][0]  # "Argmax", "Random", or "Random-Argmax"
        dynamic_q_values = ego_info[0][1]  # List of Q-values
        q_value1_acc1 = ego_info[0][1][0]
        q_value1_acc0 = ego_info[0][1][1]
        q_value1_acc_minus1 = ego_info[0][1][2]
        q_value2_acc1 = ego_info[0][1][3]
        q_value2_acc0 = ego_info[0][1][4]
        q_value2_acc_minus1 = ego_info[0][1][5]
        q_value3_acc1 = ego_info[0][1][6]
        q_value3_acc0 = ego_info[0][1][7]
        q_value3_acc_minus1 = ego_info[0][1][8]
        
        dynamic_actionget = ego_info[0][2]  # from 0 up to 8
        
        action_type = ego_info[1][0]  # ['Random-Argmax', 2] for low level
        actionget = ego_info[1][1]  # Extracted action (e.g., 2) low level


        q_values1 = ego_info[2][0][1]
        q_values2 = ego_info[2][1][1]
        q_values3 = ego_info[2][2][1]
    

        # Update ego motion and check for collisions
        ego_reached_end, ego_df, acc, merged = update_motion(ego_df=ego_df, frame_id=frame_id, act=actionget, dynamic_actionget = dynamic_actionget) #ego_df +5 rows
        state_messages = get_Message(frame_id+5, ego_df.iloc[-1], ignored_rear_vehicles=ignored_rear_vehicles, normalize=True)
        collided, ego_state, rear_v_id = collision_check(frame_id=frame_id, ego_df=ego_df, ignored_rear_vehicles=ignored_rear_vehicles)  # Check if collision occurred
        
        # Check if it's a rear-end collision that should be ignored
        if ego_state == 5:  # Rear-end collision
            # Remove the rear vehicle from the environment data
            ignored_rear_vehicles.add(rear_v_id)
            # print(f"Ignoring rear-end collision and removing vehicle ID {rear_v_id}")
            # Set collision to False to avoid terminating the episode
            collided = False
            ego_state = -1  # Set ego_state to a safe value if necessary to signify no crash

        lane_data = EPISODE_DF.loc[(EPISODE_DF['Frame_ID'] == frame_id) & (EPISODE_DF['Vehicle_ID'] == EGO_ID), 'Lane_ID'].item()
        # x_data = EPISODE_DF.loc[(EPISODE_DF['Frame_ID'] == frame_id) & (EPISODE_DF['Vehicle_ID'] == EGO_ID), 'Local_Y'].item()
        x_training = ego_df['Local_Y'].iloc[-1]
        x_data_series = EPISODE_DF.loc[(EPISODE_DF['Frame_ID'] == frame_id+5) & (EPISODE_DF['Vehicle_ID'] == EGO_ID), 'Local_Y']
        if frame_id == frames.iloc[0]:
            x_data = x_data_series.item()
            real_trajectory.append(x_data)
            simulated_trajectory.append(x_data) # ego_df['Local_Y'].iloc[0]
        elif len(x_data_series) == 1:
            x_data = x_data_series.item()
            real_trajectory.append(x_data)
            simulated_trajectory.append(x_training)

        # Calculate the reward
        reward = get_reward(crash=collided, state_messages=state_messages, 
                            ego_velocity=state_messages[6], ego_lane=state_messages[7], 
                            fc_d=state_messages[2], dist_end_merging=state_messages[8]    , 
                            x_training=x_training, x_data=x_data, lane_training=state_messages[7],
                            lane_data=lane_data, acc =acc, merged= merged,
                            real_trajectory=real_trajectory, simulated_trajectory=simulated_trajectory,
                            is_terminal = (ego_reached_end or collided))
        reward_table.append(reward)
        save_rewards([reward], file_path=f"{DATA_TRAINING_PATH}/training_history/rewards_scenario.csv", reset=False)
        total_timesteps += 1

        
        # Check if the episode should end
        if ego_reached_end:
            ego_state = 4

        # Save updated motion for the next frames
        for new_frame_id in range(frame_id + 1, frame_id + SKIP_FRAMES + 1):
            if new_frame_id >= new_episode_df[new_episode_df['Vehicle_ID'] == EGO_ID]['Frame_ID'].iloc[-1]: # handling of the last frame id 
                new_frame_id = new_episode_df[new_episode_df['Vehicle_ID'] == EGO_ID]['Frame_ID'].iloc[-1]
            df_index = new_episode_df.index[(new_episode_df['Frame_ID'] == new_frame_id) & 
                                            (new_episode_df['Vehicle_ID'] == EGO_ID)]
            ego_row = ego_df.loc[ego_df['Frame_ID'] == new_frame_id]
            new_episode_df.at[df_index[0], 'Lane_ID'] = ego_row.Lane_ID.item()
            new_episode_df.at[df_index[0], 'Local_Y'] = ego_row.Local_Y.item()
            new_episode_df.at[df_index[0], 'Mean_Speed'] = ego_row.Mean_Speed.item()
            new_episode_df.at[df_index[0], 'Mean_Accel'] = ego_row.Mean_Accel.item()

        # Ignore if the ego state is "A Car Merged into Ego"
        if ego_state != 1:
            state_messages = np.reshape(state_messages, [1, 1, Params.num_observations])
            remember_frame(currentstate = currentstate, 
                                            actionget = actionget,
                                            state_messages = state_messages,
                                            reward = reward,
                                            collided = collided,
                                            ego_reached_end = ego_reached_end,
                                            state_size = Params.num_observations,
                                            dynamic_actionget = dynamic_actionget)
            
        # Append the current information about the ego vehicle to the DataFrame
        ego_df_row = [scenario_id, (episode_no), total_timesteps - 1, ego_state] + currentstate_list + \
                    dynamic_q_values+q_values1+q_values2+q_values3+\
                     [dynamic_actionget, dynamic_action_type, actionget]
        ego_log = ego_log._append(pd.DataFrame([ego_df_row], columns=ego_log.columns), ignore_index=True)
        last_ego_row = pd.DataFrame([ego_df_row], columns=ego_log.columns)

        with open(EGO_LOG_PATH, 'a') as f:
            last_ego_row.to_csv(f, index=False, header=f.tell() == 0)  # Add header only if file is empty

        # Perform experience replay and update the model
        if total_timesteps > Params.replay_start_size and total_timesteps % 4 == 0:
            loss = DYNAMIC_AGENT.replay(Params.batch_size)
            q_loss_table.append(loss[0])

        # Update the target model every few steps
        if total_timesteps % Params.target_up == 0:
            DYNAMIC_AGENT.update_target_model()

        if ego_state == 1:
            print("==============================================")
            print("================IGNORE CRASH==================")
        if collided and first_collision:
            
            first_collision = False
            
            print(f"""
                  -- COLLISION OCCURRED
                  -- EGO IS PENALIZED BUT THE EPISODE KEEPS GOING ON.
                  """)
            
            # termination_frame_id = frame_id
            # new_episode_df = new_episode_df[new_episode_df['Frame_ID'] <= termination_frame_id]           
            # print(f"""
            # ======================================================
            # ** EGO_ID: {EGO_ID} | Run No: {run_no} | Total Episode No: {episode_no} **
            # ******************************************************
            # Collision occurred at frame id: {frame_id + SKIP_FRAMES}
            # Ego State: {Params.car0_states.get(ego_state)}
            # The original final step in frame id: {frames.iloc[-1]}
            # The original final step in y (meters): {frames_y.iloc[-1]:.2f}
            # Ego terminated the episode in y (meters): {ego_df['Local_Y'].iloc[-1]:.2f}
            # ======================================================
            # """)
            
            collision_episode_keeper.append(1)         
            
            # total_timesteps -= 1
            # counter -= 1
            break
        
        if ego_reached_end:
            break

        prev_frame_id = frame_id

    
    if not collided:
        collision_episode_keeper.append(0)         
    
    # This is the end of the episode. Record necessary calculations.
    
    # Calculate the average point-wise distance
    distance = frechet(real_trajectory, simulated_trajectory)
    # print("Average Point-Wise Distance:", distance)
    
    # Calculate Frechet Distance
    frechet_distance = frechet(real_trajectory, simulated_trajectory)
    
    if not collided:
        # Save the Frechet Distance (appends by default if reset=False)
        save_point_distances([frechet_distance], file_path=f"{FRECHET_TRAINING_PATH}/training_frechet_scenario{scenario_id}.csv", reset=False)
        
        # Save the new distance (appends by default if reset=False)
        save_point_distances([distance], file_path=f"{FRECHET_TRAINING_PATH}/training_frechet_distances.csv", reset=False)
    print(f"""
    =====================================================
    ** EGO_ID: {EGO_ID} | Run No: {run_no} | Total Episode No: {episode_no} **
    *****************************************************
    Ego completed episode on frame id: {frame_id + SKIP_FRAMES}
    Ego State: {Params.car0_states.get(ego_state)}
    The original final step in frame id: {frames.iloc[-1]}
    The original final step in y (meters): {frames_y.iloc[-1]:.2f}
    Ego completed the episode in y (meters): {ego_df['Local_Y'].iloc[-1]:.2f}
    {"Ego reached the end" if ego_reached_end else "Ego DID NOT reach the end"}
    =====================================================
    """)
    
    
    DYNAMIC_AGENT.update_temperature(step = (1.0/boltzmann_decay_end_state))
    for level, agent in DYNAMIC_AGENT.agentlevk.items():
        if hasattr(agent, "update_temperature"):
            agent.update_temperature(step=(1.0 / boltzmann_decay_end_state))

    if (run_no + 1) == ego_count:
        # Now, create directories for saving the model and target weights, but only if we are saving.
        models_dir = MODELS_DIR / f"epoch_{epoch_no}"
        target_weights_dir = TARGET_WEIGHTS_PATH
        agent_data_dir = AGENT_DATA_PATH
        
        if epoch_no == 0:
            clean_directories(MODELS_DIR, target_weights_dir, agent_data_dir)

        # Ensure directories exist only when you are going to save files.
        Path(models_dir).mkdir(parents=True, exist_ok=True)
        Path(target_weights_dir).mkdir(parents=True, exist_ok=True)
        Path(agent_data_dir).mkdir(parents=True, exist_ok=True)
        # Path(training_history_dir).mkdir(parents=True, exist_ok=True)

        # Paths for model and target weight files
        models_path = models_dir / f"model{episode_no}.h5"
        target_weights_path = target_weights_dir / f"target_weight_{episode_no}.h5"

        # Save the agent's model and target weights
        DYNAMIC_AGENT.save(models_path, run_no, target_weights_path, backup=True)

        # Save the agent's memory and configuration to a pickle file
        memory_save_path = agent_data_dir / f"agent_memory_epoch{epoch_no}_episode{episode_no}.pickle"
        DYNAMIC_AGENT.save_memory(memory_save_path, episode_no, run_no)

        config_save_path = agent_data_dir / f"agent_config_epoch{epoch_no}_episode{episode_no}.pickle"
        DYNAMIC_AGENT.save_config(total_timesteps, config_save_path, episode_no, run_no)

    final_state_df = final_state_df._append({'Run_No':run_no,
                                            'Final_State':ego_state,
                                            'Training_Final_Step':(frame_id+SKIP_FRAMES)},ignore_index=True)
    
    save_path = STUDY_RES_DIR / save_dir / f"scenario{scenario_id}-iteration{o}.csv"
    new_episode_df = new_episode_df.loc[new_episode_df['Frame_ID'] <= (frame_id + SKIP_FRAMES)]
    new_episode_df.to_csv(save_path, index=None, header=True)

    counter += 1
    return total_timesteps, final_state_df

def simulate_episode(ego_log, total_timesteps, final_state_df, episode_no, run_no, ego_count, epoch_no, scenario_id, o, save_dir=""):
    ignored_rear_vehicles = set()
    
     # Initialize properly the ego dynamic history
    ego_dyn_state = np.concatenate( [
        np.zeros((1, 1, 9)),
        DYNAMIC_HISTORIES[scenario_id].reshape(1, 1, -1) ], axis=2 )
    
    hist_action_size = int(Params.dynamic_history_action) 
    counter = 0 
    state_size = 10 
    episode_no = episode_no
    total_timesteps = total_timesteps
    actionget = 0
    new_episode_df = EPISODE_DF.copy() 

    # Get the initial state of the ego vehicle
    ego_df = EPISODE_DF.loc[EPISODE_DF['Vehicle_ID'] == EGO_ID].iloc[[0]].copy()
    ego_len = ego_df['Vehicle_Length'].item()

    frames = pd.Series(EPISODE_DF.loc[EPISODE_DF['Vehicle_ID'] == EGO_ID, 'Frame_ID'].unique()) # frame ids of ego vehicle
    frames_y = EPISODE_DF.loc[EPISODE_DF['Vehicle_ID'] == EGO_ID, 'Local_Y'] # local ys of ego vehicle

    simulated_trajectory = []
    real_trajectory = []

    first_execution = True
    prev_frame_id = frames.iloc[0]
    for frame_id in frames:

        first_collision = True

        if not (frame_id == frames.iloc[0]) and 0 < frame_id - prev_frame_id < SKIP_FRAMES:
            continue

        # Initial ego state
        if frame_id == frames.iloc[0]:
            ego_position = ego_df['Local_Y'].item()
            ego_lane = ego_df['Lane_ID'].item()
            
            currentstate_list = get_Message(frame_id=frame_id, ego_current_df=ego_df, ignored_rear_vehicles=ignored_rear_vehicles, normalize=True)  # Get current state
        else:
            ego_position = ego_df.iloc[[-1]]['Local_Y'].item()
            ego_lane = ego_df.iloc[[-1]]['Lane_ID'].item()
            currentstate_list = get_Message(frame_id=frame_id, ego_current_df=ego_df.iloc[[-1]], ignored_rear_vehicles=ignored_rear_vehicles, normalize=True)

        currentstate = np.reshape(currentstate_list, [1, 1, STATE_SIZE])

        # Check if merging is required based on ego position and lane
        if (Params.start_merging_point + ego_len) < ego_position < Params.end_merging_point and ego_lane == 0:
            remove_merging = False
        else:
            remove_merging = True

        if Params.dynamic_history_action:
            scale_action = 1.0 if not Params.scale_action else (Params.num_actions - 1.0)

            if not first_execution:
                ego_dyn_state[0, 0, STATE_SIZE:] = np.concatenate((
                ego_dyn_state[0, 0, :STATE_SIZE:].copy(), 
                [actionget / scale_action],
                ego_dyn_state[0, 0, STATE_SIZE:-state_size].copy()
            ))
            
            if first_execution:
                # for s_i in range(Params.dynamic_history, 1, -1):
                #     idx_1 = ((s_i - 1) * state_size - hist_action_size)
                #     idx_2 = s_i * state_size - 2 * hist_action_size
                #     ego_dyn_state[0, 0, idx_1:idx_2] = currentstate[0, 0, :].copy()
                first_execution = False

        else:
            ego_dyn_state[0, 0, STATE_SIZE:] = ego_dyn_state[0, 0, :-STATE_SIZE:].copy()

        # Always update the first STATE_SIZE values with current state
        ego_dyn_state[0, 0, 0:STATE_SIZE] = currentstate[0, 0, :].copy()

        # Update currentstate with the dynamic state
        currentstate = ego_dyn_state.copy()

        # Get the action for the ego vehicle from the dynamic agent
        ego_info = DYNAMIC_AGENT.act(state=currentstate, remove_merging=remove_merging, get_qvals=True)

        # Extract action information
        dynamic_action_type = ego_info[0][0]  # "Argmax", "Random", or "Random-Argmax"
        dynamic_q_values = ego_info[0][1]  # List of Q-values 
        q_value1_acc1 = ego_info[0][1][0]
        q_value1_acc0 = ego_info[0][1][1]
        q_value1_acc_minus1 = ego_info[0][1][2]
        q_value2_acc1 = ego_info[0][1][3]
        q_value2_acc0 = ego_info[0][1][4]
        q_value2_acc_minus1 = ego_info[0][1][5]
        q_value3_acc1 = ego_info[0][1][6]
        q_value3_acc0 = ego_info[0][1][7]
        q_value3_acc_minus1 = ego_info[0][1][8]
        
        dynamic_actionget = ego_info[0][2]  # from 0 up to 8
        
        action_type = ego_info[1][0]  # ['Random-Argmax', 2] for low level
        actionget = ego_info[1][1]  # Extracted action (e.g., 2) low level


        q_values1 = ego_info[2][0][1]
        q_values2 = ego_info[2][1][1]
        q_values3 = ego_info[2][2][1]

        # Update ego motion and check for collisions
        ego_reached_end, ego_df, acc, merged = update_motion(ego_df=ego_df, frame_id=frame_id, act=actionget, dynamic_actionget = dynamic_actionget) #ego_df +5 rows
        state_messages = get_Message(frame_id+5, ego_df.iloc[-1], ignored_rear_vehicles=ignored_rear_vehicles, normalize=True)
        collided, ego_state, rear_v_id = collision_check(frame_id=frame_id, ego_df=ego_df, ignored_rear_vehicles=ignored_rear_vehicles)  # Check if collision occurred
        
        # Check if it's a rear-end collision that should be ignored
        if ego_state == 5:  # Rear-end collision
            # Remove the rear vehicle from the environment data
            ignored_rear_vehicles.add(rear_v_id)
            # print(f"Ignoring rear-end collision and removing vehicle ID {rear_v_id}")
            # Set collision to False to avoid terminating the episode
            collided = False
            ego_state = -1  # Set ego_state to a safe value if necessary to signify no crash

        lane_data = EPISODE_DF.loc[(EPISODE_DF['Frame_ID'] == frame_id) & (EPISODE_DF['Vehicle_ID'] == EGO_ID), 'Lane_ID'].item()
        # x_data = EPISODE_DF.loc[(EPISODE_DF['Frame_ID'] == frame_id) & (EPISODE_DF['Vehicle_ID'] == EGO_ID), 'Local_Y'].item()
        x_training = ego_df['Local_Y'].iloc[-1]
        x_data_series = EPISODE_DF.loc[(EPISODE_DF['Frame_ID'] == frame_id+5) & (EPISODE_DF['Vehicle_ID'] == EGO_ID), 'Local_Y']
        if frame_id == frames.iloc[0]:
            x_data = x_data_series.item()
            real_trajectory.append(x_data)
            simulated_trajectory.append(x_data) # ego_df['Local_Y'].iloc[0]
        elif len(x_data_series) == 1:
            x_data = x_data_series.item()
            real_trajectory.append(x_data)
            simulated_trajectory.append(x_training)
        else:
            x_data = frames_y.iloc[-1]
            real_trajectory.append(x_data)
            simulated_trajectory.append(x_training)

        # Calculate the reward
        reward = get_reward(crash=collided, state_messages=state_messages, 
                            ego_velocity=state_messages[6], ego_lane=state_messages[7], 
                            fc_d=state_messages[2], dist_end_merging=state_messages[8]    , 
                            x_training=x_training, x_data=x_data, lane_training=state_messages[7], lane_data=lane_data, acc =acc, merged= merged, real_trajectory=real_trajectory, simulated_trajectory=simulated_trajectory)
        reward_table.append(reward)
        save_rewards([reward], file_path=f"{SIMULATION_HISTORY_PATH}/rewards_scenario.csv", reset=False)
        total_timesteps += 1 

        # Check if the episode should end
        if ego_reached_end:
            ego_state = 4

        # Save updated motion for the next frames
        for new_frame_id in range(frame_id + 1, frame_id + SKIP_FRAMES + 1):
            if new_frame_id >= new_episode_df[new_episode_df['Vehicle_ID'] == EGO_ID]['Frame_ID'].iloc[-1]: # handling of the last frame id
                new_frame_id = new_episode_df[new_episode_df['Vehicle_ID'] == EGO_ID]['Frame_ID'].iloc[-1]
            df_index = new_episode_df.index[(new_episode_df['Frame_ID'] == new_frame_id) & 
                                            (new_episode_df['Vehicle_ID'] == EGO_ID)]
            ego_row = ego_df.loc[ego_df['Frame_ID'] == new_frame_id]
            new_episode_df.at[df_index[0], 'Lane_ID'] = ego_row.Lane_ID.item()
            new_episode_df.at[df_index[0], 'Local_Y'] = ego_row.Local_Y.item()
            new_episode_df.at[df_index[0], 'Mean_Speed'] = ego_row.Mean_Speed.item()
            new_episode_df.at[df_index[0], 'Mean_Accel'] = ego_row.Mean_Accel.item()

         # Append the current information about the ego vehicle to the DataFrame
        ego_df_row = [scenario_id, (episode_no), total_timesteps - 1, ego_state] + currentstate_list + \
                    dynamic_q_values+q_values1+q_values2+q_values3+\
                     [dynamic_actionget, dynamic_action_type, actionget]
        ego_log = ego_log._append(pd.DataFrame([ego_df_row], columns=ego_log.columns), ignore_index=True)
        last_ego_row = pd.DataFrame([ego_df_row], columns=ego_log.columns)

        with open(EGO_LOG_PATH, 'a') as f:
            last_ego_row.to_csv(f, index=False, header=f.tell() == 0)  # Add header only if file is empty
        
        if ego_state == 1:
            print("==============================================")
            print("================IGNORE CRASH==================")
        
        if collided and first_collision:
            
            first_collision = False
            
            print(f"""
                  -- COLLISION OCCURRED
                  -- EGO IS PENALIZED BUT THE EPISODE KEEPS GOING ON.
                  """)
            
            collision_episode_keeper.append(1)  

            break

        # If collided or reached the end, stop the simulation
        if ego_reached_end: #if collided or ego_reached_end:
            break

        prev_frame_id = frame_id

    if not collided:
        collision_episode_keeper.append(0)         
    
    # This is the end of the episode. Record necessary calculations.
    
    # Calculate Frechet Distance
    frechet_distance = frechet(real_trajectory, simulated_trajectory)

    if not collided:
        # Save the Frechet Distance (appends by default if reset=False)
        save_point_distances([frechet_distance], file_path=f"{FRECHET_SIMULATION_PATH}/simulation_frechet_scenario{scenario_id}.csv", reset=False)
        
        # Save the new distance (appends by default if reset=False)
        save_point_distances([frechet_distance], file_path=f"{FRECHET_SIMULATION_PATH}/simulation_frechet_distances.csv", reset=False)
    print(f"""
    =====================================================
    ** EGO_ID: {EGO_ID} | Run No: {run_no} | Total Episode No: {episode_no} **
    *****************************************************
    Ego completed episode on frame id: {frame_id + SKIP_FRAMES}
    Ego State: {Params.car0_states.get(ego_state)}
    The original final step in frame id: {frames.iloc[-1]}
    The original final step in y (meters): {frames_y.iloc[-1]:.2f}
    Ego completed the episode in y (meters): {ego_df['Local_Y'].iloc[-1]:.2f}
    {"Ego reached the end" if ego_reached_end else "Ego DID NOT reach the end"}
    =====================================================
    """)
    

    final_state_df = final_state_df._append({'Run_No':run_no,
                                            'Final_State':ego_state,
                                            'Training_Final_Step':(frame_id+SKIP_FRAMES)},ignore_index=True)
    
    save_path = STUDY_RES_DIR / save_dir / f"scenario{scenario_id}-iteration{o}.csv"
    new_episode_df = new_episode_df.loc[new_episode_df['Frame_ID'] <= (frame_id + SKIP_FRAMES)]
    new_episode_df.to_csv(save_path, index=None, header=True)

    counter += 1
    return total_timesteps, final_state_df

def clean_directories(*directories):
    """Clean multiple directories passed as arguments."""
    for idx, dir_path in enumerate(directories):
        if dir_path.exists():
            for item in dir_path.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            print(f"Cleaned directory {idx + 1}: {dir_path}")
        else:
            print(f"No directory {idx + 1} found: {dir_path}")

def plot_graphs():

    plt.figure(2)
    plt.plot(moving_average(reward_table, 1000), color = "red") # 500
    plt.title("Training:Reward vs Step") 
    plt.ylabel("Average Reward")
    plt.xlabel("Step")
    reward_save_path = DATA_SIMULATION_PATH / "reward_vs_step.png" if Simulate else TRAINING_HISTORY_PATH / "reward_vs_step.png"
    plt.savefig(reward_save_path)
    plt.close()

        # Create a DataFrame to store the values
    
    # Plot Q-Loss over training episodes
    if Train:
        plt.figure(3)
        plt.plot(moving_average(q_loss_table, 10), color="green")  #100
        plt.title("Training: Q-Loss vs Step")
        plt.ylabel("Q-Loss (MSE)")
        plt.xlabel("Episode")
        q_loss_save_path = TRAINING_HISTORY_PATH / "q_loss_vs_episode.png"
        plt.savefig(q_loss_save_path)
        plt.close()
    
    window_size = 10
    collision = moving_average(collision_episode_keeper, window_size)
    plt.figure(1)
    plt.plot(collision, color = "blue" )
    plt.title("Training: Collision vs Episode") 
    plt.ylabel("Average Collision Value per Episode")
    plt.xlabel("Episode")
    collision_save_path = DATA_SIMULATION_PATH / "collision_vs_episode.png" if Simulate else TRAINING_HISTORY_PATH / "collision_vs_episode.png"
    plt.savefig(collision_save_path)
    plt.close()
        
def save_point_distances(new_distances, file_path="point_distances.csv", reset=False):
    """
    Save point distances to a CSV file, starting fresh if reset is True.
    
    Parameters:
    - new_distances: list of float, distances to save.
    - file_path: str, path to the CSV file.
    - reset: bool, if True, overwrites the file; otherwise, appends.
    """
    import pandas as pd

    # Convert the distances to a DataFrame
    df = pd.DataFrame(new_distances, columns=["Point Distance"])

    if reset:
        # Overwrite the file with new data
        df.to_csv(file_path, index=False)
    else:
        # Append to the existing file
        df.to_csv(file_path, mode='a', index=False, header=not os.path.exists(file_path))

def save_rewards(rewards, file_path="step_reward.csv", reset=False):
    """
    Save point distances to a CSV file, starting fresh if reset is True.
    
    Parameters:
    - new_distances: list of float, distances to save.
    - file_path: str, path to the CSV file.
    - reset: bool, if True, overwrites the file; otherwise, appends.
    """
    import pandas as pd

    # Convert the distances to a DataFrame
    df = pd.DataFrame(rewards, columns=["Reward"])

    if reset:
        # Overwrite the file with new data
        df.to_csv(file_path, index=False)
    else:
        # Append to the existing file
        df.to_csv(file_path, mode='a', index=False, header=not os.path.exists(file_path))

def plot_point_distances(point_distances, window_size=100):
    # Calculate running average
    running_avg = [sum(point_distances[max(0, i - window_size):i]) / min(i, window_size)
                   for i in range(1, len(point_distances) + 1)]

    # Plot both raw Frechet distances and the running average
    plt.figure(figsize=(10, 5))
    plt.plot(point_distances, color="blue", label="Frechet Distance (Raw)")
    plt.plot(running_avg, color="red", label=f"Running Average")
    plt.title("Frechet Distance vs. Episode")
    plt.xlabel("Episode")
    plt.ylabel("Frechet Distance")
    plt.legend()
    plt.grid(True)
    plt.savefig("frechet_distance_vs_episode.png")
    # plt.show()

def main():
    num_epochs = NUM_EPOCHS

    ego_df_columns = EGO_DF_DYN_COLS
    ego_log = pd.DataFrame(columns=ego_df_columns)
    episode_no = 0

    if Train:
        select_path()
        load_dynamic_model()
        if os.path.exists(EGO_LOG_PATH):
            os.remove(EGO_LOG_PATH)  # Remove the file if it already exists

        pd.DataFrame(columns=EGO_DF_DYN_COLS).to_csv(EGO_LOG_PATH, index=False)  # Write header
    if Simulate:
        select_path(is_simulation=True)
        SIMULATION_HISTORY_PATH.mkdir(parents=True, exist_ok=True)
        EGO_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        load_dynamic_model(is_simulation=True)
        pd.DataFrame(columns=EGO_DF_DYN_COLS).to_csv(EGO_LOG_PATH, index=False)  # Write header


    output_path = Path(STUDY_RES_DIR) / "epochs"
    clean_directories(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    total_timesteps = 0 

    if Train:
        directories = Path(TIMED_DATA_PATH) / "train_scenarios.csv"
    else:
        directories = Path(TIMED_DATA_PATH) / "test_scenarios.csv"
        
    scenario_ids_df = pd.read_csv(directories)
    scenario_id_list = scenario_ids_df['Scenarios'].tolist()

    global DYNAMIC_HISTORIES
    # Load dynamic histories from the pickle file
    with EGO_DYNAMIC_HISTORIES_PATH.open('rb') as file:
        DYNAMIC_HISTORIES = pickle.load(file)
        DYNAMIC_HISTORIES = np.array(list(DYNAMIC_HISTORIES.values()))
        # You access the initial dynamic history of the scenario N with DYNAMIC_HISTORIES[N]
    
    # At the very beginning of your script or `main` function
    if Train:
        for run_no, scenario_id in enumerate(scenario_id_list):
            save_point_distances([], file_path=f"{FRECHET_TRAINING_PATH}/training_frechet_scenario{scenario_id}.csv", reset=True)
    elif Simulate:
        for run_no, scenario_id in enumerate(scenario_id_list):
            save_point_distances([], file_path=f"{FRECHET_SIMULATION_PATH}/simulation_frechet_scenario{scenario_id}.csv", reset=True)

    for epoch_no in range(num_epochs):
        print(f"*Starting epoch no: {epoch_no}**")
        # Select ego id 
        for run_no, scenario_id in enumerate(scenario_id_list):

            for o in range(SCENARIO_ITERATION):
                load_episode_df(scenario_id = int(scenario_id))
                save_dir = Path("epochs") / f"epoch_{epoch_no}"

                if Train:
                    final_state_df = pd.DataFrame(columns=FINAL_STATE_DF_COLUMNS_TRAINING)
                    Path(STUDY_RES_DIR / save_dir).mkdir(parents=True, exist_ok=True)
                    (total_timesteps, final_state_df) = run_episode(ego_log,
                                                    total_timesteps,
                                                    save_dir = Path(save_dir), 
                                                    final_state_df = final_state_df,
                                                    run_no = run_no,
                                                    ego_count = len(scenario_id_list),
                                                    episode_no = episode_no,
                                                    epoch_no=epoch_no,
                                                    scenario_id=scenario_id,
                                                    o = o)
                    episode_no += 1
                
                    # Save the final state dataframe
                    final_state_df.to_csv(STUDY_RES_DIR / save_dir / "final_state.csv", index=None, header=True)

                    del final_state_df

                if Simulate:
                    final_state_df = pd.DataFrame(columns=FINAL_STATE_DF_COLUMNS_SIMULATION)  
                    Path(STUDY_RES_DIR / save_dir).mkdir(parents=True, exist_ok=True)

                    (total_timesteps, final_state_df) = simulate_episode(ego_log,
                        total_timesteps,
                        save_dir=Path(save_dir),
                        final_state_df=final_state_df,
                        run_no = run_no,
                        ego_count = len(scenario_id_list),
                        episode_no = episode_no,
                        epoch_no=epoch_no,
                        scenario_id=scenario_id,
                        o = o)
                    episode_no += 1

                    # Save the final state dataframe for simulation
                    final_state_df.to_csv(STUDY_RES_DIR / save_dir / "sim_final_state.csv", index=None, header=True)
                    del final_state_df

    plot_graphs()

if __name__ == '__main__':
    main()