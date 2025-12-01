"""
Simulation script to observe a dynamic agent replacing a vehicle which is selected
and recorded in vs_always6_final_filtered or vs_startsAt7_endsAt6_final_filtered
Two types of situations are simulated: no_followers and with_followers
These indicate that vehicles that stay behind the ego on the same lane are ignored
in order to prevent these crashing into the ego from the rear

Steps:
Make sure quadruplet data exists on base / data / NGSIM_I80_quadruplets
Set MODEL_LEVEL1, MODEL_LEVEL2, MODEL_LEVEL3, MODEL_DYNAMIC model numbers. These are related to level-k models.
Set FOLLOWER_SETTING, EGO_INIT_LANE to your preference
Do Training.
Before doing simulation, choose MODEL_DATA and EPOCH_NO depending on where the latest model of the data training belongs to.
Increment your STUDY_TITLE if you want a fresh start without deleting the current study. 

"""
from Params import Params
from src.DynamicDQNAgent import DynamicDQNAgent
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import warnings  
import shutil
import pickle
warnings.simplefilter(action='ignore', category=FutureWarning) # not to see futurewarning warning

Simulate = True
Train = False
MODEL_DATA = 314 # The model from data that will be used # for simulation
EPOCH_NO = "epoch_6" # which env no the model above belongs to # for simulation
NUM_EPOCHS = 5
SCENARIO_ITERATION = 5


"""
MODEL_LEVEL1 = 99 # Model number of the level1 DQN
MODEL_LEVEL2 = 99 # Model number of the level2 DQN
MODEL_LEVEL3 = 99 # Model number of the level3 DQN
MODEL_DYNAMIC = 99  # Model number of the dynamic DQN
"""

SAMPLING_FREQ = 10 
SKIP_FRAMES = int(Params.timestep * 10)
SECTION_DUR = 10

BASE_DIR = Path(__file__).parent.resolve() # Root path for your ATR Data
DATA_PATH = BASE_DIR / "data"

TIMED_DATA_PATH = DATA_PATH / "5sec_scenarios"
TIMED_SCENARIOS_PATH = TIMED_DATA_PATH / "scenarios"
SCENARIO_HISTORIES_PATH = TIMED_DATA_PATH / "histories"
EGO_DYNAMIC_HISTORIES_PATH = TIMED_DATA_PATH / "dynamic_histories.pickle"

QUADRUPLETS_PATH = DATA_PATH / "NGSIM_I80_quadruplets"
NGSIM_RESULTS_PATH = BASE_DIR / "NGSIM_I80_Results"

STUDY_TITLE = "study_data_IDM" # you can change this.
STUDY_PATH = NGSIM_RESULTS_PATH / STUDY_TITLE
DATA_SIMULATION_PATH = STUDY_PATH / "simulation"
FRECHET_SIMULATION_PATH = Path(DATA_SIMULATION_PATH) / "frechet_histories"
FRECHET_SIMULATION_PATH.mkdir(parents=True, exist_ok=True)


"""
AGENT_DATA_PATH = STUDY_PATH / "training" / "agent_data"
MODELS_DIR = STUDY_PATH / "training" / "models"
TARGET_WEIGHTS_PATH = STUDY_PATH / "training" / "target_weights"
TRAINING_HISTORY_PATH = STUDY_PATH / "training" / "training_history"
"""
SIMULATION_HISTORY_PATH = STUDY_PATH / "simulation" / "simulation_history"

ENABLE_BOLTZMANN_SAMPLING = True 
ENABLE_DYNAMIC_DRIVING_BOLTZMANN_SAMPLING = True
ENABLE_RANDOM_DYNAMIC_STRATEGY = False # the code is set up for always False case

EGO_DF_DYN_COLS = ['Scenario_ID','Episode', 'Time_Step', 'State', 'fs_d', 'fs_v', 'fc_d', 
                   'fc_v', 'rs_d', 'rs_v', 'velocity', 'lane', 'dist_end_merging', 
                    'Action']
#EGO_LOG_PATH = TRAINING_HISTORY_PATH / "ego_log.csv"

#FINAL_STATE_DF_COLUMNS_TRAINING = ['Run_No','Final_State','Real_Final_Step','Training_Final_Step']
FINAL_STATE_DF_COLUMNS_SIMULATION = ['Run_No','Final_State','Real_Final_Step','Simulation_Final_Step']

STATE_SIZE = Params.num_observations
ACTION_SIZE = Params.num_actions
"""
DYNAMIC_STATE_SIZE = Params.num_observations * Params.dynamic_history + \
    Params.dynamic_history_action * (Params.dynamic_history-1)
DYNAMIC_ACTION_SIZE = Params.num_dynamic_actions
"""

IDM_MOBIL_AGENT = None
EPISODE_DF = None
EGO_ID = None

# For the 5 second scenario implementation
DYNAMIC_HISTORIES = None

# EGO_IDs = pd.read_csv(DATA_PATH / "ego_ids" / "vs_always6_final_filtered.csv")
# EGO_IDs = pd.concat([EGO_IDs, pd.read_csv(DATA_PATH / "ego_ids" / "vs_startsAt7_endsAt6_final_filtered.csv")], axis=0)

collision_episode_keeper = []
#q_loss_table = [] 
reward_table = []
point_distances = []


class IDM_MOBIL():
    # Intelligent Driver Model with Minimizing Overall Braking Induced by Lane change
    def __init__(self, v_0=Params.nominal_speed, T=1.5, s_0= 2 , a=1.5, b=3): # params from Kesting, Treiber, and Helbing 
        self.a = a  # Maximum acceleration
        self.T = T  # Safe time headway 
        self.s_0 = s_0  # Minimum gap (front bumper to rear bumper)
        self.v_0 = v_0  # Desired velocity
        self.b = b  # Comfortable deceleration
        self.car_length = Params.carlength
    
    def IDM_acc(self, v, dv, d):
        """
        Calculate acceleration using the Intelligent Driver Model.
        
        Args:
            v: Current velocity of the ego vehicle
            dv: Relative velocity (lead vehicle minus ego vehicle)
            d: Gap distance (bumper to bumper)
            
        Returns:
            Acceleration value
        s_0 is  Minimum gap (front bumper to rear bumper)
        T: Safe time headway 
        """
        s = max(0.1, d - self.car_length)  # Ensure minimum positive gap
        s_star = self.s_0 + max(0, v*self.T + (v*dv)/(2 * (self.a*self.b)**0.5))
        dvdt = self.a * (1 - (v/self.v_0)**4 - (s_star/s)**2)
        
        return dvdt
    
    def act(self, state_messages):
        """
        An adapted version of the IDM-MOBIL model of Yaren's, already including MOBIL_lane_change's functionality.
        """

        fs_d = state_messages[0]  # Front car distance
        fs_v = state_messages[1]  # Front car velocity
        fc_d = state_messages[2]        
        fc_v = state_messages[3]
        rs_d = state_messages[4]  # Rear car distance
        rs_v = state_messages[5]  # Rear car velocity
        ego_v = state_messages[6] 
        ego_lane = state_messages[7]
        dist_end_merging = state_messages[8]
        acc = self.IDM_acc(v = ego_v, dv = -fc_v, d = fc_d)
        lane_change = 0

    # Consider merging if on ramp and inside merging region
        if ego_lane == 0 and Params.start_merging_point < dist_end_merging < Params.end_merging_point:
            if fs_d > self.s_0 and rs_d > self.s_0:
                # Compute accelerations in the target lane
                new_acc = self.IDM_acc(v=ego_v, dv=fs_v, d=fs_d)
                following_acc = self.IDM_acc(v=ego_v - rs_v, dv=-rs_v, d=rs_d)

                # MOBIL-like decision
                if new_acc > acc - 0.2 and following_acc > -self.b / 2:
                    lane_change = 1  # merge to lane 1
        
        return acc, lane_change


def load_idm_mobil_model():
    """
    Create an instance of the IDM_MOBIL agent.
    """
    global IDM_MOBIL_AGENT
    
    # Initialize the IDM_MOBIL agent with default parameters
    IDM_MOBIL_AGENT = IDM_MOBIL(
        v_0=Params.nominal_speed,  # Desired velocity
        T=1.5,                     # Safe time headway
        s_0=Params.close_distance, # Minimum gap
        a=2.5,                     # Maximum acceleration
        b=3                        # Comfortable deceleration
    )
    
    print("IDM_MOBIL agent initialized with default parameters")






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

def average_pointwise_distance(p, q):
    """
    Compute the average point-wise distance between two trajectories.

    Parameters:
    - p: list or numpy array, trajectory 1
    - q: list or numpy array, trajectory 2

    Returns:
    - float: average point-wise distance
    """
    # Ensure both trajectories are numpy arrays
    p = np.array(p)
    q = np.array(q)

    # Check if trajectories have the same length
    if len(p) != len(q):
        raise ValueError("The two trajectories must have the same length.")

    # Compute absolute differences and calculate the mean
    avg_distance = np.mean(np.abs(p - q))

    return avg_distance


def moving_average(data, window_size):
    """Compute the moving average with a specified window size."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def get_real_final_step():
    real_final_step = EPISODE_DF.loc[(EPISODE_DF['Local_Y']<=Params.end_for_car0)&(
        EPISODE_DF['Vehicle_ID']==EGO_ID)].iloc[-1].Frame_ID.item()
    return real_final_step

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


#Normalize state variables
def not_normalize_state(state):
    msg = []
    msg.append(state[0]) #fs_d
    msg.append(state[1]) #fs_v
    msg.append(state[2]) #fc_d
    msg.append(state[3]) #fc_v
    msg.append(state[4]) #rs_d
    msg.append(state[5]) #rs_v
    msg.append(state[6]) #vel
    msg.append(float(state[7])) #lane
    msg.append(state[8]) #dist_end_merging
    return msg

#def remember_frame(currentstate, actionget,  state_messages, reward, collided, #I DELETED THIS

def get_reward(crash, state_messages, ego_velocity, ego_lane, fc_d, dist_end_merging, x_training, x_data, lane_training, lane_data, acc, merged, real_trajectory, simulated_trajectory):
    performance = 1.0
    scale = 1 # 0.02
    
    wc = 1000 * scale # 1000 # Collision
    wv = 10 * scale * performance # Velocity
    we = 5 * scale #*performance # Effort
    wh = 5 * scale # Headway
    wnm = 50 * scale #*performance # Not Merging
    ws = 30 * scale # 100  #  *performance # Velocity Less than 2.25m/s or Stopping on Lane-0 with dist_end_merging less than far distance
    w7 = 15 #TODO tune
    w8 = -150 #TODO tune

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
    if ego_lane == 1:  # Ego is on the main road
        # Penalize if the ego is too far behind, not accelerating, or decelerating unnecessarily
        if acc >= 1.25 and fc_d >= Params.far_distance and (dist_end_merging >= Params.far_distance or dist_end_merging <= 0):
            s = 1 
            # print("Main road: Penalize for not accelerating")
        elif acc <= -1.1 and fc_d >= Params.far_distance and (dist_end_merging >= Params.far_distance or dist_end_merging <= 0):
            s = -1 # Penalize for not accelerating in safe conditions
        else:
            s = 0  # No penalty if conditions are met
            # print("Main road: No penalty")

    # Ramp Case
    elif ego_lane == 0:  # Ego is on the ramp
        # Inside the merging region
        if dist_end_merging < Params.merging_region_length:
            # Penalize if not merging while it's safe
            if merged != 5:
                if fl_d >= Params.close_distance and abs(rl_d) >= (1.5 * Params.far_distance):
                    s = -1  # Penalize for not merging when safe
                    # print("Ramp: Penalize for not merging in safe conditions")
                elif dist_end_merging <= Params.far_distance:
                    s = -0.05  # Small penalty for stopping too close to the merging zone end
                    # print("Ramp: Small penalty for being close to merging zone end")
            else:
                s = 0  # No penalty for merging
                # print("Ramp: No penalty, merging safely")

        # Outside the merging region
        else:
            if ego_velocity < -Params.hard_decel_rate*Params.timestep and fc_d >= Params.far_distance:
                s = -1  # Penalize for not accelerating or preparing to merge
                # print("Ramp: Penalize for not preparing to merge")


    ox = 0
    if len(simulated_trajectory) >= 2 or len(real_trajectory) >= 2:
        frechet_prev = abs(simulated_trajectory[-2] - real_trajectory[-2])
        frechet_now = abs(simulated_trajectory[-1] - real_trajectory[-1])

        ox = (frechet_prev-frechet_now)



    # Calibration term for y-axis (lane offset oy)
    oy = 1 if int(lane_training) != lane_data else 0

    # Final reward with calibration terms
    R = wc * c + wv * v + we * e + wh * h + wnm * nm + ws * s
    R_calib = R + ox * w7 + oy * w8 

    return R_calib


def update_motion(ego_df, frame_id, state_messages):
    """
    Update the motion of the ego vehicle using the IDM_MOBIL model.
    
    Args:
        ego_df: DataFrame containing ego vehicle information
        frame_id: Current frame ID
        state_messages: State information about surrounding vehicles (normalized)
        
    Returns:
        tuple: (reached_end, updated_ego_df, acceleration, merged)
    """
    ego_class_id = ego_df['Vehicle_Class_ID'].iloc[-1]
    ego_len = ego_df['Vehicle_Length'].iloc[-1]
    ego_follower_id = ego_df['Follower_ID'].iloc[-1]
    ego_leader_id = ego_df['Leader_ID'].iloc[-1]
    ego_pos = ego_df['Local_Y'].iloc[-1]    
    ego_lane = ego_df['Lane_ID'].iloc[-1]
    ego_vel = ego_df['Mean_Speed'].iloc[-1]
    
    # Get acceleration and lane change decision from IDM_MOBIL agent
    acc, lane_change = IDM_MOBIL_AGENT.act(state_messages)
    
    # Apply acceleration limits, IMPORTANT
    acc = max(min(acc, Params.actions[3][1]), Params.actions[4][1])
    
    timestep = Params.timestep/SKIP_FRAMES
    merged = False
    
    new_lane = ego_lane
    new_vel = ego_vel
    new_pos = ego_pos
    
    # Apply lane change if determined by IDM_MOBIL
    if lane_change == 1 and ego_lane == 0:
        new_lane += 1
        merged = True
    
    # Update position and velocity for each frame
    for idx, new_frame in enumerate(range(frame_id+1, frame_id+SKIP_FRAMES+1)):
        new_vel += acc * timestep
        
        # Apply velocity limits
        if new_vel < Params.min_speed:
            new_vel = Params.min_speed
        elif new_vel > Params.max_speed:
            new_vel = Params.max_speed
    
        # Update position based on velocity
        if new_vel == Params.min_speed:
            new_pos += (ego_vel + 0.5*(Params.min_speed-ego_vel))*timestep
        elif new_vel == Params.max_speed:
            new_pos += (ego_vel + 0.5*(Params.max_speed-ego_vel))*timestep
        else:
            new_pos += (ego_vel+0.5*(acc*timestep))*timestep
            
        temp_lane = ego_lane
        if idx == SKIP_FRAMES - 1:            
            temp_lane = new_lane

        # Add new row to ego_df
        ego_df = ego_df._append({
            "Vehicle_ID": EGO_ID,
            "Frame_ID": new_frame,
            "Lane_ID": temp_lane,
            "Local_Y": new_pos,
            "Mean_Speed": new_vel,
            "Mean_Accel": acc,
            "Vehicle_Length": ego_len,
            "Vehicle_Class_ID": ego_class_id, 
            "Follower_ID": ego_follower_id,
            "Leader_ID": ego_leader_id
        }, ignore_index=True)
        
        ego_vel = new_vel
        
    # Check if the ego has passed the end of the road
    reached_end = new_pos > Params.end_for_car0
    
    return reached_end, ego_df, acc, merged, new_lane


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
        msg = not_normalize_state(msg)
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


def translate_car0_state(ego_state):
    return Params.car0_states[ego_state]


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


def simulate_episode(ego_log, total_timesteps, final_state_df, episode_no, run_no, ego_count, epoch_no, scenario_id, save_dir=""):

    # Initialize a set to keep track of ignored rear vehicles
    ignored_rear_vehicles = set()
    simulated_trajectory = []
    real_trajectory = []

    episode_no = episode_no
    total_timesteps = total_timesteps

    new_episode_df = EPISODE_DF.copy() # these are already converted to SI

    # Get the initial state of the ego vehicle
    ego_df = EPISODE_DF.loc[EPISODE_DF['Vehicle_ID'] == EGO_ID].iloc[[0]].copy()
    ego_len = ego_df['Vehicle_Length'].item()
    
    frames = EPISODE_DF[EPISODE_DF.duplicated(subset='Frame_ID', keep='first') == False].Frame_ID
    frames_y = EPISODE_DF.loc[EPISODE_DF['Vehicle_ID'] == EGO_ID, 'Local_Y']

    prev_frame_id = frames.iloc[0]
    for frame_id in frames:
      
        # Skip frames that don't meet the SKIP_FRAMES criteria
        if not (frame_id == frames.iloc[0]) and 0 < frame_id - prev_frame_id < SKIP_FRAMES:
            continue

        # Initial ego state
        x_data_series = EPISODE_DF.loc[(EPISODE_DF['Frame_ID'] == frame_id+5) & (EPISODE_DF['Vehicle_ID'] == EGO_ID), 'Local_Y']
        
        # Initial ego state
        if frame_id == frames.iloc[0]:
            ego_position = ego_df['Local_Y'].item()
            ego_lane = ego_df['Lane_ID'].item()
            x_data = x_data_series.item()
            real_trajectory.append(x_data)
            simulated_trajectory.append(x_data)

            currentstate_list = get_Message(frame_id=frame_id, ego_current_df=ego_df, 
                                           ignored_rear_vehicles=ignored_rear_vehicles, normalize=True)
        else:
            ego_position = ego_df.iloc[[-1]]['Local_Y'].item()
            ego_lane = ego_df.iloc[[-1]]['Lane_ID'].item()
            currentstate_list = get_Message(frame_id=frame_id, ego_current_df=ego_df.iloc[[-1]], 
                                           ignored_rear_vehicles=ignored_rear_vehicles, normalize=True)
            if not x_data_series.empty:
                x_data = x_data_series.item()
            real_trajectory.append(x_data)
            simulated_trajectory.append(ego_df['Local_Y'].iloc[-1])
        currentstate = np.reshape(currentstate_list, [1, 1, STATE_SIZE])

        # Update ego motion using IDM_MOBIL
        ego_reached_end, ego_df, acc, merged, lane = update_motion(ego_df=ego_df, frame_id=frame_id, state_messages=currentstate_list)
        
        # Get the next state
        state_messages = get_Message(frame_id+SKIP_FRAMES, ego_df.iloc[-1], 
                                    ignored_rear_vehicles=ignored_rear_vehicles, normalize=True)
        
        # Check for collisions
        collided, ego_state, rear_v_id = collision_check(frame_id=frame_id, ego_df=ego_df, 
                                                         ignored_rear_vehicles=ignored_rear_vehicles)
        
        # Check if it's a rear-end collision that should be ignored
        if ego_state == 5:  # Rear-end collision
            ignored_rear_vehicles.add(rear_v_id)
            collided = False
            ego_state = -1

        # # Get the original vehicle trajectory data
        # lane_data = EPISODE_DF.loc[(EPISODE_DF['Frame_ID'] == frame_id) & 
        #                            (EPISODE_DF['Vehicle_ID'] == EGO_ID), 'Lane_ID'].item()
        # x_training = ego_df['Local_Y'].iloc[-1]
        # x_data_series = EPISODE_DF.loc[(EPISODE_DF['Frame_ID'] == frame_id+SKIP_FRAMES) & 
        #                                (EPISODE_DF['Vehicle_ID'] == EGO_ID), 'Local_Y']
        
        # # Record trajectories for comparison
        # if frame_id == frames.iloc[0]:
        #     x_data = x_data_series.item() if not x_data_series.empty else frames_y.iloc[-1]
        #     real_trajectory.append(x_data)
        #     simulated_trajectory.append(x_data)
        # elif not x_data_series.empty:
        #     x_data = x_data_series.item()
        #     real_trajectory.append(x_data)
        #     simulated_trajectory.append(x_training)
        # else:
        #     x_data = frames_y.iloc[-1]
        #     real_trajectory.append(x_data)
        #     simulated_trajectory.append(x_training)

        # Calculate the reward
        reward = get_reward(crash=collided, state_messages=state_messages,
                            ego_velocity=state_messages[6], ego_lane=state_messages[7],
                            fc_d=state_messages[2], dist_end_merging=state_messages[8],
                            x_training=ego_df['Local_Y'].iloc[-1], x_data=state_messages[0], 
                            lane_training=state_messages[7], lane_data=state_messages[7], acc = acc, merged = merged, real_trajectory = real_trajectory, simulated_trajectory = simulated_trajectory)

        reward_table.append(reward)
        save_rewards([reward], file_path=f"{SIMULATION_HISTORY_PATH}/rewards_scenario.csv", reset=False)
        total_timesteps += 1

        # Check if the episode should end
        if not collided and ego_reached_end:
            ego_state = 4

        # Save updated motion for the next frames
        for new_frame_id in range(frame_id+1, frame_id+SKIP_FRAMES+1):
            if new_frame_id >= new_episode_df[new_episode_df['Vehicle_ID'] == EGO_ID]['Frame_ID'].iloc[-1]:
                new_frame_id = new_episode_df[new_episode_df['Vehicle_ID'] == EGO_ID]['Frame_ID'].iloc[-1]
            
            df_index = new_episode_df.index[(new_episode_df['Frame_ID'] == new_frame_id) & 
                                           (new_episode_df['Vehicle_ID'] == EGO_ID)]
            if len(df_index) > 0:  # Check if indices exist
                ego_row = ego_df.loc[ego_df['Frame_ID'] == new_frame_id]
                if not ego_row.empty:
                    new_episode_df.at[df_index[0], 'Lane_ID'] = ego_row.Lane_ID.item()
                    new_episode_df.at[df_index[0], 'Local_Y'] = ego_row.Local_Y.item()
                    new_episode_df.at[df_index[0], 'Mean_Speed'] = ego_row.Mean_Speed.item()
                    new_episode_df.at[df_index[0], 'Mean_Accel'] = ego_row.Mean_Accel.item()

        if merged:
            action = 5  # Already merged
        else:
            if acc > Params.hard_accel_rate - 0.5:  # Strong acceleration (> 2.5)
                action = 3  # Hard accelerate
            elif acc > Params.accel_rate - 0.5:  # Moderate acceleration (> 1.5)
                action = 1  # Normal accelerate
            elif acc < Params.hard_decel_rate + 0.5:  # Strong braking (< -4.0)
                action = 4  # Hard brake
            elif acc < Params.decel_rate + 0.5:  # Moderate braking (< -1.5)
                action = 2  # Normal brake
            else:  # Small acceleration/deceleration (-1.5 to 1.5)
                action = 0  # Maintain speed
        # Append the current information about the ego vehicle to the DataFrame
        ego_df_row = [scenario_id, episode_no, total_timesteps-1, ego_state] + currentstate_list + [action]
        ego_log = ego_log._append(pd.DataFrame([ego_df_row], columns=ego_log.columns), ignore_index=True)

        # If collided or reached the end, stop the simulation
        if collided or ego_reached_end:
            break

        prev_frame_id = frame_id

    if not collided:
        collision_episode_keeper.append(0)  
    
    
    frechet_distance = frechet(real_trajectory, simulated_trajectory)
    save_point_distances([frechet_distance], file_path=f"{FRECHET_SIMULATION_PATH}/simulation_frechet_scenario{scenario_id}.csv", reset=False)
    save_point_distances([frechet_distance], file_path=f"{FRECHET_SIMULATION_PATH}/simulation_frechet_distances.csv", reset=False)

           
    
    # This is the end of the episode. Record necessary calculations.
    
    # Calculate the average point-wise distance
    distance = average_pointwise_distance(real_trajectory, simulated_trajectory)
    print("Average Point-Wise Distance:", distance)
    
    # Calculate Frechet Distance
    

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
    
    # Record final state
    final_state_df = final_state_df._append({
        'Run_No': run_no,
        'Final_State': ego_state,
        'Real_Final_Step': get_real_final_step(),
        'Simulation_Final_Step': (frame_id + SKIP_FRAMES)
    }, ignore_index=True)
    
    # Save the results
    save_path = DATA_SIMULATION_PATH / save_dir / f"{scenario_id}.csv"
    Path(save_path.parent).mkdir(parents=True, exist_ok=True)
    new_episode_df = new_episode_df.loc[new_episode_df['Frame_ID'] <= (frame_id + SKIP_FRAMES)]
    new_episode_df.to_csv(save_path, index=None, header=True)

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
    """
    Plot reward and collision graphs from the simulation results.
    """
    plt.figure(2)
    plt.plot(moving_average(reward_table, 100), color="red")
    plt.title("IDM_MOBIL: Reward vs Step") 
    plt.ylabel("Average Reward")
    plt.xlabel("Step")
    reward_save_path = DATA_SIMULATION_PATH / "reward_vs_step_IDM_Mobile.png"
    plt.savefig(reward_save_path)
    plt.close()
    
    window_size = 10
    collision = moving_average(collision_episode_keeper, window_size)
    plt.figure(1)
    plt.plot(collision, color="blue")
    plt.title("IDM_MOBIL: Collision vs Episode") 
    plt.ylabel("Average Collision Value per Episode")
    plt.xlabel("Episode")
    collision_save_path = DATA_SIMULATION_PATH / "collision_vs_episode.png"
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
    """
    Main function to run the IDM_MOBIL simulation.
    """
    num_epochs = NUM_EPOCHS

    # Initialize logging DataFrame
    ego_df_columns = EGO_DF_DYN_COLS
    ego_log = pd.DataFrame(columns=ego_df_columns)
    episode_no = 0

    # Select output path and initialize IDM_MOBIL agent
    load_idm_mobil_model()

    # Create output directory
    output_path = Path(DATA_SIMULATION_PATH) / "simulation_history"
    clean_directories(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    total_timesteps = 0

    # Load scenario IDs
    directories = Path(TIMED_DATA_PATH) / "test_scenarios.csv"
    scenario_ids_df = pd.read_csv(directories)
    scenario_id_list = scenario_ids_df['Scenarios'].tolist()

    # Reset distance tracking files
    for scenario_id in scenario_id_list:
        save_point_distances([], file_path=f"{FRECHET_SIMULATION_PATH}/simulation_frechet_distances.csv", reset=True)
        save_point_distances([], file_path=f"{FRECHET_SIMULATION_PATH}/simulation_frechet_scenario{scenario_id}.csv", reset=True)



    for run_no, scenario_id in enumerate(scenario_id_list):
            # Load episode data
            load_episode_df(scenario_id=int(scenario_id))
            save_dir = Path("simulation_history") / f"scenario_id_{scenario_id}"
            
            # Initialize final state dataframe
            final_state_df = pd.DataFrame(columns=FINAL_STATE_DF_COLUMNS_SIMULATION)
            Path(DATA_SIMULATION_PATH / save_dir).mkdir(parents=True, exist_ok=True)

            # Run simulation
            (total_timesteps, final_state_df) = simulate_episode(
                ego_log,
                total_timesteps,
                save_dir=Path(save_dir),
                final_state_df=final_state_df,
                run_no=run_no,
                ego_count=len(scenario_id_list),
                episode_no=episode_no,
                epoch_no=1,
                scenario_id=scenario_id,
            )
            episode_no += 1

            # Save final state information
            final_state_df.to_csv(DATA_SIMULATION_PATH / save_dir / "sim_final_state.csv", index=None, header=True)
            del final_state_df

    plot_graphs()
    # plot_point_distances(point_distances, window_size=100)


if __name__ == '__main__':
    main()