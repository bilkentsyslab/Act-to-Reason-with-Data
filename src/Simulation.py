#This file implements the simulation of level-k/dynamic agents in highway merging
# © 2020 Cevahir Köprülü All Rights Reserved

from Params import Params
from src.Message import Message
from src.DQNAgent import DQNAgent
from src.State import State
from src.DynamicDQNAgent import DynamicDQNAgent
from src.SimulationAnalyzer import SimulationAnalyzer
import numpy as np
import os 
import pandas as pd
import random
import math
from pathlib import Path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} ) 
# sess = tf.Session(config=config) 
# keras.backend.set_session(sess)

SIM_DF_COLS = ['Episode', 'Time_Step', 'State', 'Car_ID', 'Lane_ID', 
               'Position_X', 'Velocity_X', 'Acceleration_X',
               'Level_k', 'Action', 'Dynamic_Action']

EGO_DF_DYN_COLS = ['Episode', 'Time_Step', 'State', 'fs_d', 'fs_v', 'fc_d', 
                   'fc_v', 'rs_d', 'rs_v', 'velocity', 'lane', 'dist_end_merging', 
                   'q-lev1', 'q-lev2', 'q-lev3', 'q-maintain1', 'q-accel1', 
                   'q-decel1', 'q-hard_accel1', 'q-hard_decel1', 'q-merge1',
                   'q-maintain2', 'q-accel2', 'q-decel2', 'q-hard_accel2', 
                   'q-hard_decel2', 'q-merge2', 'q-maintain3', 'q-accel3', 
                   'q-decel3', 'q-hard_accel3', 'q-hard_decel3', 'q-merge3', 
                   'Dynamic_Action', 'Dynamic_Action_Type', 'Action', 
                   'Acceleration_X', 'Reward']

EGO_DF_DYN_RND_COLS = ['Episode', 'Time_Step', 'State', 'fs_d', 'fs_v', 'fc_d', 
                       'fc_v', 'rs_d', 'rs_v', 'velocity', 'lane', 'dist_end_merging', 
                       'q-maintain1', 'q-accel1', 'q-decel1', 'q-hard_accel1', 
                       'q-hard_decel1', 'q-merge1', 'q-maintain2', 'q-accel2', 
                       'q-decel2', 'q-hard_accel2', 'q-hard_decel2', 'q-merge2', 
                       'q-maintain3', 'q-accel3', 'q-decel3', 'q-hard_accel3', 
                       'q-hard_decel3', 'q-merge3', 'Dynamic_Action', 
                       'Dynamic_Action_Type', 'Action', 'Acceleration_X', 'Reward']

EGO_DF_LEVK_COLS = ['Episode', 'Time_Step', 'State', 'fl_d', 'fl_v', 'fc_d', 
                    'fc_v', 'rs_d', 'rs_v', 'velocity', 'lane', 
                    'dist_end_merging', 'q-maintain', 'q-accel', 'q-decel', 
                    'q-hard_accel', 'q-hard_decel', 'q-merge', 'Action', 
                    'Action_Type', 'Acceleration_X', 'Reward']

EGO_DF_LEV0_COLS = ['Episode', 'Time_Step', 'State', 'fl_d', 'fl_v', 'fc_d', 
                    'fc_v', 'rs_d', 'rs_v', 'velocity', 'lane', 
                    'dist_end_merging', 'Action']   

COL_DF_COLS = ['Episode', 'numcars', 'Time_Step', 'fs_d', 'fs_v', 'fc_d', 
               'fc_v', 'rs_d', 'rs_v', 'velocity', 'lane', 'dist_end_merging', 
               'Action', 'State', 'Level_k']

class Simulation():
    
    def __init__(self, levelk_config, simulation_config, file_config):
        self.levelk_config = levelk_config
        self.file_config = file_config
        self.simulation_config = simulation_config
        self.state_size = Params.num_observations
        self.action_size = Params.num_actions
        self.dynamic_state_size = Params.num_observations * Params.dynamic_history + \
            Params.dynamic_history_action * (Params.dynamic_history-1)
        self.dynamic_action_size = Params.num_dynamic_actions       
        self.models = self._load_models()
        self.files, self.analysis_files = self._create_files()

    def _load_models(self):
        """
        Load required models to be used during a simulation session

        Returns
        -------
        models : dictionary
            Keeps DQNAgent and DynamicDQNAgent objects.

        """
        
        models = {1:None,2:None,3:None,4:None}
        agent_levelk_paths = {1:None,2:None,3:None}
        for i in range(1,4):
            if self.levelk_config["models"][i] != None:
                agent_levelk_paths[i] = self.file_config["path"]+"level"+str(i)+\
                                          "\\training\\models\\model"+\
                                          str(self.levelk_config["models"][i])
                
        if self.levelk_config["vs"] == -1:
            for i in range(1,4):
                agentlevk = DQNAgent(self.state_size, self.action_size) # Level-k Agent
                agentlevk.load(agent_levelk_paths[i]) # Loads the model of trained level-k agent
                agentlevk.T = agentlevk.MIN_T #Sets the boltzmann temp. of Level-k cars to 1, prevents random actions   
                agentlevk.boltzmann_sampling = self.levelk_config["boltzmann_sampling"]                 
                models[i] = agentlevk

        elif 1 <= self.levelk_config["vs"] <= 3:
            agentlevk = DQNAgent(self.state_size, self.action_size) # Level-k Agent
            agentlevk.load(agent_levelk_paths[self.levelk_config["vs"]]) # Loads the model of trained level-k agent
            agentlevk.T = agentlevk.MIN_T #Sets the boltzmann temp. of Level-k cars to 1, prevents random actions
            agentlevk.boltzmann_sampling = self.levelk_config["boltzmann_sampling"]
            models[self.levelk_config["vs"]] = agentlevk
        
        elif self.levelk_config["vs"] == 4:
            levelk_config = {"paths": agent_levelk_paths, "boltzmann_sampling": self.levelk_config["dynamic_driving_boltzmann_sampling"]}
            dynamic_agent = DynamicDQNAgent(self.dynamic_state_size,
                                            self.dynamic_action_size,  
                                            self.state_size, self.action_size,
                                            levelk_config)
            if not self.levelk_config["random_dynamic_strategy"]:
                dynamic_agent.load(self.file_config["path"]+"dynamic/training/models/model"+\
                                   str(self.levelk_config["models"][4]))
            dynamic_agent.T = dynamic_agent.MIN_T  
            dynamic_agent.boltzmann_sampling = self.levelk_config["boltzmann_sampling"]
            dynamic_agent.random_dynamic_strategy = self.levelk_config["random_dynamic_strategy"]
            models[4] = dynamic_agent    
            
        if self.levelk_config["ego"] != 0:
            if models[self.levelk_config["ego"]] == None:
                if 1 <= self.levelk_config["ego"] <= 3:
                    agentlevk = DQNAgent(self.state_size, self.action_size) # Level-k Agent
                    agentlevk.load(agent_levelk_paths[self.levelk_config["ego"]]) # Loads the model of trained level-k agent
                    agentlevk.T = agentlevk.MIN_T #Sets the boltzmann temp. of Level-k cars to prevent random actions  
                    agentlevk.boltzmann_sampling = self.levelk_config["boltzmann_sampling"]
                    models[self.levelk_config["ego"]] = agentlevk  
                else:
                    levelk_config = {"paths": agent_levelk_paths, "boltzmann_sampling": self.levelk_config["dynamic_driving_boltzmann_sampling"]}
                    dynamic_agent = DynamicDQNAgent(self.dynamic_state_size,
                                                    self.dynamic_action_size, 
                                                    self.state_size, 
                                                    self.action_size,
                                                    levelk_config)
                    if not self.levelk_config["random_dynamic_strategy"]:
                        dynamic_agent.load(self.file_config["path"]+\
                                           "dynamic/training/models/model"+\
                                               str(self.levelk_config["models"][4]))
                    dynamic_agent.T = dynamic_agent.MIN_T
                    dynamic_agent.boltzmann_sampling = self.levelk_config["boltzmann_sampling"]
                    dynamic_agent.random_dynamic_strategy = self.levelk_config["random_dynamic_strategy"]
                    models[4] = dynamic_agent          
            
        return models

    def _create_files(self):
        """
        Create paths for simulation/collision/ego dataframe files and also for
        SimulationAnalyzer output files
        
        Returns
        -------
        files : dictionary
            A dictionary of pathways for dataframes used to record simulation related information.
        analysis_files : dictionary
            A dictionary of pathways to save SimulationAnalyzer outputs.

        """
        
        path = self.file_config["path"]
        directory = self.file_config["directory"]
        Path("./"+path+directory).mkdir(parents=True, exist_ok=True)
        
        files = {"sim_df":None, 
                 "col_df":None, 
                 "ego_df":None}
        
        analysis_files = {"analysis_df_file":None,
                          "ego_df_analysis_file":None,
                          "succ_eps_pickle":None,
                          "long_eps_pickle":None,
                          "crash_eps_pickle":None}
        
        num_episodes = self.simulation_config["num_episodes"]

        if self.levelk_config["ego"] < 4:
            generic_file = path + directory + "simulation_level_"+\
                str(self.levelk_config["ego"])
            
            if self.levelk_config["ego"] > 0:
                generic_file = generic_file + "_m"+ \
                    str(self.levelk_config["models"][self.levelk_config["ego"]])
        else:
            if self.levelk_config["random_dynamic_strategy"]:
                generic_file = path + directory + "simulation_dynamic_random"             
            else:
                generic_file = path + directory + "simulation_dynamic_m"+\
                    str(self.levelk_config["models"][self.levelk_config["ego"]])
        
        if 0 == self.levelk_config["vs"]:
            generic_file = generic_file + "_vs_L"+str(self.levelk_config["vs"])+\
                "_"+str(len(self.simulation_config["car_population"]))+ "x"+\
                    str(num_episodes)+"eps"
        
        elif 1 <= self.levelk_config["vs"] <= 3:
            generic_file = generic_file + "_vs_L"+str(self.levelk_config["vs"])+\
                "_m"+str(str(self.levelk_config["models"][self.levelk_config["vs"]]))+\
                    "_"+str(len(self.simulation_config["car_population"]))+"x"+\
                        str(num_episodes)+"eps"
        elif self.levelk_config["vs"] == 4:
            if self.levelk_config["random_dynamic_strategy"]:
                generic_file = generic_file + "_vs_dynamic"+"_random_"+\
                        str(len(self.simulation_config["car_population"]))+"x"+\
                            str(num_episodes)+"eps"    
            else:
                generic_file = generic_file + "_vs_dynamic"+"_m"+\
                    str(self.levelk_config["models"][self.levelk_config["vs"]])+"_"+\
                        str(len(self.simulation_config["car_population"]))+"x"+\
                            str(num_episodes)+"eps"
        else:
            vs_extension = "_vs_L0_and_L1_m"+str(str(self.levelk_config["models"][1]))
            vs_extension += "_and_L2_m"+str(str(self.levelk_config["models"][2]))
            vs_extension += "_and_L3_m"+str(str(self.levelk_config["models"][3]))
            generic_file = generic_file + vs_extension + "_"+\
                str(len(self.simulation_config["car_population"]))+"x"+\
                    str(num_episodes)+"eps"        

        files["col_df"] = generic_file + "_col.csv"
        if os.path.exists(files["col_df"]):
            os.remove(files["col_df"])      
        
        if self.file_config["save_sim_df"]:
            files["sim_df"] = generic_file + "_sim.csv"
            if os.path.exists(files["sim_df"]):
                os.remove(files["sim_df"])             
            
            analysis_files["succ_eps_pickle"] = generic_file + "_successful_eps.pickle"
            analysis_files["long_eps_pickle"] = generic_file + "_long_eps.pickle"
            analysis_files["crash_eps_pickle"] = generic_file + "_crash_eps.pickle"
        
        if self.file_config["save_ego_df"]:
            analysis_files["ego_df_analysis_file"] = generic_file + "_ego_analysis.csv"
            files["ego_df"] = generic_file + "_ego.csv"
            if os.path.exists(files["ego_df"]):
               os.remove(files["ego_df"]) 
           
        analysis_files["analysis_df_file"] = generic_file + "_analysis.csv"
               
        return files, analysis_files
    
    def randomize_levelk_distribution(self):
        """
        Randomly sample the ratio of level-k vehicles in an episode

        Returns
        -------
        levelk_config : dictionary
            Includes ego and other vehicle level-k setting.

        """
        levelk_config = {"ego": self.levelk_config["ego"],
                         "others": None}
        
        dist_bound = self.levelk_config["distribution_bound"]
        
        l0 = math.exp(random.uniform(-dist_bound,dist_bound))
        l1 = math.exp(random.uniform(-dist_bound,dist_bound))
        l2 = math.exp(random.uniform(-dist_bound,dist_bound))
        l3 = math.exp(random.uniform(-dist_bound,dist_bound))
        total_prob = l0+l1+l2+l3
        
        l0 = l0/total_prob
        l1 = l1/total_prob
        l2 = l2/total_prob
        l3 = l3/total_prob
        print("Distribution: L0="+str(round(l0,2))+" L1=" + str(round(l1,2))+
              " L2=" + str(round(l2,2))+" L3=" + str(round(l3,2)))
        
        levelk_config["others"] = np.array([l0, l1, l2, l3, 0.0])    
        return levelk_config
    
    def get_state_action(self, state, dyn_msg):
        """
        This function gets actions of level-k agents in batches
        Merging and non-merging vehicles are processed separately

        Parameters
        ----------
        state : State object
            State of an episode.
        dyn_msg : dictionary
            Keeps the history of dynamic agents.

        Returns
        -------
        cars_lev0 : list
            IDs of level-0 vehicles.
        acts : numpy array
            Actions of level-1,2,3 vehicles.
        dyn_acts : numpy array
            Actions of dynamic vehicles.
        dyn_msg : dictionary
            Keeps the history of dynamic agents.

        """
        
        # States dictionary to keep states of the vehicles
        # Outer keys refer to the level-k strategy
        # Inner keys refer to whether a vehicle can merge or not
        # 0: No merging
        # 1: Merging
        states = {1: {0:[], 1:[]},
                  2: {0:[], 1:[]},
                  3: {0:[], 1:[]},
                  4: {0:[], 1:[]}}
        # Car dictionary to keep IDs of the vehicles
        # Outer keys refer to the level-k strategy
        # Inner keys refer to whether a vehicle can merge or not
        # 0: No merging
        # 1: Merging
        cars = {0: [],
                1: {0:[], 1:[]},
                2: {0:[], 1:[]},
                3: {0:[], 1:[]},
                4: {0:[], 1:[]}}
        
        hist_action_size = int(Params.dynamic_history_action)
        state_size = self.state_size + hist_action_size
        
        # Iterate through every vehicle except the ego vehicle
        for idx_vehicle in range(1,state.cars_info.shape[1]):
            levk = state.cars_info2[5,idx_vehicle]
            # Append the state of a level-0 agent as a Message objects
            if levk == 0:
                cars[levk].append(state.get_Message(idx_vehicle))            
            else: # For others, append them to their respective lists
                currentstate_temp = state.get_Message(idx_vehicle, normalize=True) #state of the ego car
                currentstate_temp = np.reshape(currentstate_temp, [1, self.state_size])
                
                if (Params.start_merging_point + Params.carlength < state.cars_info[0,idx_vehicle] < 
                    Params.end_merging_point) and state.cars_info2[0,idx_vehicle] == 0:
                    merging = True                    
                else:
                    merging = False
 
                cars[levk][int(merging)].append(idx_vehicle)
                if 1 <= levk <= 3:
                    states[levk][int(merging)].append(currentstate_temp)
                else:
                    state_temp = np.zeros((1,self.dynamic_state_size))
                    state_temp[0,0:self.state_size] = currentstate_temp[0,:].copy()
                    
                    # If $dynamic_history$ is more than 1, stack the current frame
                    for t_i in range(1,Params.dynamic_history): #1 to 10
                        idx_1 = (t_i)*state_size-hist_action_size
                        if idx_vehicle in dyn_msg[t_i]: # Vehicle was in the environment t_i frames ago
                            idx_2 = (t_i+1)*state_size-hist_action_size
                            state_temp[0,idx_1:idx_2] = dyn_msg[t_i][idx_vehicle].copy()
                        else: # Vehicle was not in the environment t_i frames ago
                            idx_2 = (t_i+1)*state_size-2*hist_action_size
                            state_temp[0,idx_1:idx_2] = currentstate_temp[0,:].copy()
                            if Params.dynamic_history_action:
                                dyn_msg[t_i][idx_vehicle] = np.concatenate((currentstate_temp[0,:].copy(),[0]))
                            else:
                                dyn_msg[t_i][idx_vehicle] = currentstate_temp[0,:].copy()
                                                                        
                    states[levk][int(merging)].append(state_temp) 
                                                      
        acts = np.zeros((state.cars_info.shape[1]))
        dyn_acts = np.zeros((state.cars_info.shape[1]))

        # Select actions of Level-k agents by Boltzmann Sampling
        
        # Level-1: Merging
        if cars[1][1]:
            acts[np.asarray(cars[1][1])] = self.models[1].act_inbatch(np.asarray(states[1][1]),
                                                                      remove_merging=False)
        # Level-1: No Merging
        if cars[1][0]:
            acts[np.asarray(cars[1][0])] = self.models[1].act_inbatch(np.asarray(states[1][0]),
                                                                      remove_merging=True)
        # Level-2: Merging
        if cars[2][1]:
            acts[np.asarray(cars[2][1])] = self.models[2].act_inbatch(np.asarray(states[2][1]),
                                                                      remove_merging=False)
        # Level-2: No Merging
        if cars[2][0]:
            acts[np.asarray(cars[2][0])] = self.models[2].act_inbatch(np.asarray(states[2][0]),
                                                                      remove_merging=True)
        # Level-3: Merging
        if cars[3][1]:
            acts[np.asarray(cars[3][1])] = self.models[3].act_inbatch(np.asarray(states[3][1]),
                                                                      remove_merging=False)
        # Level-3: No Merging
        if cars[3][0]:
            acts[np.asarray(cars[3][0])] = self.models[3].act_inbatch(np.asarray(states[3][0]),
                                                                      remove_merging=True) 
        # Dynamic: Merging
        if cars[4][1]:
            acts[ np.asarray(cars[4][1])], dyn_acts[np.asarray(cars[4][1])]  = self.models[4].act_inbatch(
                np.asarray(states[4][1]),remove_merging=False)
        # Dynamic: No Merging
        if cars[4][0]:
            acts[np.asarray(cars[4][0])], dyn_acts[
                np.asarray(cars[4][0])] = self.models[4].act_inbatch(
                    np.asarray(states[4][0]),remove_merging=True)    
   
        # Update dyn_msg dictionary
        for t_i in range(Params.dynamic_history-1,0,-1):
            if len(dyn_msg[t_i]) != 0:
                if t_i == 1:
                    for i,dyn_id in enumerate(cars[4][0]):
                        if Params.dynamic_history_action:
                            scale_action = 1.0 if not Params.scale_action else (Params.num_actions-1.0)
                            dyn_msg[t_i][dyn_id] = np.concatenate((
                                states[4][0][i][0,0:self.state_size].copy(),
                                [acts[dyn_id]/scale_action]))
                        else:
                            dyn_msg[t_i][dyn_id] = states[4][0][i][0,0:self.state_size].copy()
                    for i,dyn_id in enumerate(cars[4][1]):
                        if Params.dynamic_history_action:
                            scale_action = 1.0 if not Params.scale_action else (Params.num_actions-1.0)
                            dyn_msg[t_i][dyn_id] = np.concatenate((
                                states[4][1][i][0,0:self.state_size].copy(),
                                [acts[dyn_id]/scale_action]))
                        else:
                            dyn_msg[t_i][dyn_id] = states[4][1][i][0,0:self.state_size].copy()
                else:
                    dyn_msg[t_i] = dyn_msg[t_i-1].copy()
                
        return cars[0], acts, dyn_acts, dyn_msg
    
    def run(self):
        """
        Runs the simulation

        Returns
        -------
        None.

        """
        ego_lane = self.simulation_config["ego_lane"]
        add_car_prob = self.simulation_config["add_car_prob"]
        bias_lane0vs1 = self.simulation_config["bias_lane0vs1"]
        maxcars_onlane0 = self.simulation_config["maxcars_onlane0"]
        car_population = self.simulation_config["car_population"]
        max_episode_length = self.simulation_config["max_episode_length"]
        num_episodes = self.simulation_config["num_episodes"]
        
        # Number of episodes = num_state_resets*num_episodes
        num_state_resets = len(car_population)

        # Level-k agent distribution in the environment
        # 0: Level-0
        # 1: Level-1
        # 2: Level-2
        # 3: Level-3
        # 4: Dynamic
        distribution = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        distribution[self.levelk_config["vs"]] = 1.0
        levelk_config = {"ego": self.levelk_config["ego"],
                         "others": distribution}
         
        if self.file_config["save_ego_df"]:
            if self.levelk_config["ego"] == 4:
                if self.levelk_config["random_dynamic_strategy"]:
                    ego_df_columns = EGO_DF_DYN_RND_COLS
                else:
                    ego_df_columns = EGO_DF_DYN_COLS
            elif 1 <= self.levelk_config["ego"] <= 3:
                ego_df_columns = EGO_DF_LEVK_COLS
            else:
                ego_df_columns = EGO_DF_LEV0_COLS   

        done = False
        collision_count = 0
        collision_type1_count = 0
        long_episodes = 0
        for state_no in range (0, num_state_resets):
            print('State ' + str(state_no) + ' with ' + str(car_population[state_no]) + ' cars')
        
            for episode_no in range(0, num_episodes):
                numcars = car_population[state_no]

                print('Episode ' + str(num_episodes*state_no+episode_no))
                
                ego_reached_end = False
                counter = 0 # Steps taken in an episode
                
                if self.levelk_config["vs"] == -1:
                    levelk_config = self.randomize_levelk_distribution()                    

                # Initializes the environment
                state = State(numcars, ego_lane, bias_lane0vs1, 
                                    maxcars_onlane0, levelk_config)
                numcars = state.cars_info.shape[1]
                print("Numcars: "+str(numcars))
                
                # List to keep the status of every vehicle
                # 1: Vehicle is in the environment
                # 0: Vehicle passed the end of the environment
                cars_status = [1 for temp_i in range(numcars)]       
        
                # Initilize dynamic agent properties
                dyn_msg = {}
                for dyn_msg_i in range(1,Params.dynamic_history):
                    dyn_msg[dyn_msg_i] = {}
                if self.levelk_config["ego"] == 4:
                    ego_dyn_state = np.zeros((1,1,self.dynamic_state_size))
                    hist_action_size = int(Params.dynamic_history_action)
                    state_size = self.state_size + hist_action_size
                    actionget = 0

                ego_state = -1 # Check the car0_states dictionary in Params.py
                
                # Initialize simulation dataframe and append the initial frame
                if self.file_config["save_sim_df"]:
                    sim_df = pd.DataFrame(columns= SIM_DF_COLS);
                    
                    for temp_idx in range(state.cars_info.shape[1]):
                        temp_state = "N/A"
                        if temp_idx == 0:
                            temp_state = ego_state
                        sim_df = sim_df._append({"Episode":(num_episodes*state_no+episode_no),
                                                "Time_Step":counter,
                                                "State":temp_state,"Car_ID":temp_idx,
                                                "Lane_ID":state.cars_info2[0,temp_idx],
                                                "Position_X":state.cars_info[0,temp_idx],
                                                "Velocity_X":state.cars_info[2,temp_idx],
                                                "Acceleration_X":state.cars_info[3,temp_idx],
                                                "Level_k":state.cars_info2[5,temp_idx], 
                                                "Action": state.cars_info2[1,temp_idx], 
                                                "Dynamic_Action":state.cars_info2[5,temp_idx]},
                                               ignore_index=True);
                 
                    with open(self.files["sim_df"],'a') as f:                
                        sim_df.to_csv(f, index=None, header=f.tell()==0)
                        del sim_df
                
                # Initialize ego dataframe
                if self.file_config["save_ego_df"]:
                   ego_df = pd.DataFrame(columns=ego_df_columns);             
                
                while not ego_reached_end:
                    # Earlist list is used to add a new vehicle
                    # [Position on the ramp, Speed on the ramp, 
                    # earliest_vehicle_pos_main_road, earliest_vehicle_speed_main_road]
                    earliest = [Params.init_size,Params.max_speed,Params.init_size,Params.max_speed]
                    
                    # Initialize simulation dataframe
                    if self.file_config["save_sim_df"]:
                        sim_df = pd.DataFrame(columns = SIM_DF_COLS);
              
                    if self.levelk_config["ego"] != 0:
                        # State/Observation of the ego car
                        currentstate_list = state.get_Message(0,normalize=True) 
                        currentstate = np.reshape(currentstate_list, [1, 1, self.state_size])
                        
                        # Dynamic ego section
                        if self.levelk_config["ego"] == 4:
                            if Params.dynamic_history_action:
                                scale_action = 1.0 if not Params.scale_action else (Params.num_actions-1.0)
                                ego_dyn_state[0,0,self.state_size:] = np.concatenate((
                                    ego_dyn_state[0, 0, :self.state_size].copy(),[actionget/scale_action],
                                    ego_dyn_state[0,0,self.state_size:-state_size].copy()))
                            else:
                                ego_dyn_state[0,0,self.state_size:] = ego_dyn_state[0,0,:-self.state_size].copy()
                            ego_dyn_state[0,0,0:self.state_size] = currentstate[0,0,:].copy()
                            if counter == 0:
                                for s_i in range(Params.dynamic_history,1,-1):
                                    idx_1 = ((s_i-1)*state_size-hist_action_size)
                                    idx_2 = s_i*state_size-2*hist_action_size
                                    ego_dyn_state[0,0,idx_1:idx_2] = currentstate[0,0,:].copy()
    
                            currentstate = ego_dyn_state.copy()
                        
                        # Determine whether merging should be available for the ego
                        if ((Params.start_merging_point + Params.carlength) < 
                            state.cars_info[0,0] < Params.end_merging_point and state.cars_info2[0,0] == 0):
                            remove_merging = False
                        else:
                            remove_merging = True
                            
                        # Get ego related info if ego or crash history dataframes 
                        # are recorded 

                        #                            
                        if self.file_config["save_ego_df"]:
                            ego_info = self.models[self.levelk_config["ego"]].act(
                                state=currentstate, 
                                remove_merging=remove_merging, 
                                get_qvals=True) # Action taken by the ego car
                            
                            if self.levelk_config["ego"] == 4:
                                # Dynamic Model Output
                                if self.levelk_config["random_dynamic_strategy"]:
                                    dynamic_action_type = ego_info[0][0]
                                    dynamic_actionget = ego_info[0][1]
                                else:
                                    dynamic_action_type = ego_info[0][0]
                                    dynamic_q_values = ego_info[0][1]
                                    dynamic_actionget = ego_info[0][2]                                  
                                # Level-1 Model Output
                                q_values1 = ego_info[1][0][1]
                                # Level-2 Model Output
                                q_values2 = ego_info[1][1][1]
                                # Level-3 Model Output
                                q_values3 = ego_info[1][2][1]
                                # Driving Action
                                action_type = ego_info[1][dynamic_actionget][0]
                                actionget = ego_info[1][dynamic_actionget][2]
                            elif 1 <= self.levelk_config["ego"] <= 3:
                                action_type = ego_info[0]
                                q_values = ego_info[1]
                                actionget = ego_info[2]                                
                        else:
                            ego_info = self.models[self.levelk_config["ego"]].act(
                                state=currentstate, 
                                remove_merging=remove_merging, 
                                get_qvals=False) # Action taken by the ego car
                            
                            if self.levelk_config["ego"] == 4:
                                dynamic_action_type = ego_info[0][0]
                                dynamic_actionget = ego_info[0][1]
                                action_type = ego_info[1][0]
                                actionget = ego_info[1][1]
                            elif 1 <= self.levelk_config["ego"] <= 3:
                                action_type = ego_info[0]
                                actionget = ego_info[1]
                    else:
                        currentstate_list = state.get_Message(0,normalize=False) #state of the ego car
                                
                    cars_lev0, acts, dyn_acts, dyn_msg = self.get_state_action(state,dyn_msg)

                    counter += 1
                    del_cars = []
                    cars_lev0_counter = 0
                    for temp_idx in range(state.cars_info.shape[1]):
                        if temp_idx == 0:
                            # Update the position of the ego car based on the selected action
                            if state.cars_info2[5,temp_idx] == 0:
                                ego_reached_end = state.update_motion(car_id=temp_idx, 
                                                                      msg=Message(currentstate_list))
                                currentstate_list = state.normalize_state(currentstate_list)
                            else:
                                ego_reached_end = state.update_motion(car_id = temp_idx, 
                                                                      act = actionget)
                                if state.cars_info2[5,temp_idx] == 4:
                                    dyn_acts[0] = dynamic_actionget
                        else:
                            # Update the position of a surrounding vehicle
                            if state.cars_info2[5,temp_idx] == 0:
                                temp_reached_end = state.update_motion(car_id=temp_idx, 
                                                                       msg=Message(cars_lev0[cars_lev0_counter]))
                                cars_lev0_counter += 1
                            else:
                                temp_reached_end = state.update_motion(car_id=temp_idx, 
                                                                       act=acts[temp_idx])
                            
                            # If a vehicle reaches the end and its status is not updated,
                            # update its status to be used for adding a new vehicle
                            if temp_reached_end and cars_status[temp_idx] == 1:
                                state.cars_info2[5,temp_idx] = 0
                                del_cars.append(temp_idx)
                                cars_status[temp_idx] = 0
                                    
                        if self.file_config["save_sim_df"]:    
                            sim_df = sim_df._append({"Episode":(num_episodes*state_no+episode_no),
                                                    "Time_Step":counter,
                                                    "State": ego_state,
                                                    "Car_ID": temp_idx,
                                                    "Lane_ID":state.cars_info2[0,temp_idx],
                                                    "Position_X":state.cars_info[0,temp_idx],
                                                    "Velocity_X":state.cars_info[2,temp_idx],
                                                    "Acceleration_X":state.cars_info[3,temp_idx],
                                                    "Level_k":state.cars_info2[5,temp_idx], 
                                                    "Action":state.cars_info2[1,temp_idx],
                                                    "Dynamic_Action":dyn_acts[temp_idx]+1},ignore_index=True);
                        
                        # Update earlist list to add a new vehicle appropriately
                        if state.cars_info[0,temp_idx] < earliest[0 + 2*state.cars_info2[0,temp_idx]]:
                            earliest[0 + 2*state.cars_info2[0,temp_idx]] = state.cars_info[0,temp_idx]
                            earliest[1 + 2*state.cars_info2[0,temp_idx]] = state.cars_info[2,temp_idx]
                            
                    state.check_reset_positions() # Reset vehicle positions if a collision occurs
                    nextstate = state.get_Message(0,normalize=True) # State reached by the ego car
                    [done,ego_state] = state.check_ego_state(ignore_stopping=True) # Checks if there is a collision or not
                    reward = state.get_reward(done,nextstate) # Current reward

                    # Assign the state as "Ego reached the end" if it has done so
                    if not done and ego_reached_end:
                        ego_state = 4
                        
                    # Change the state of the ego vehicle in the simulation dataframe
                    if self.file_config["save_sim_df"]:
                        sim_df.iloc[-state.cars_info.shape[1],2] = ego_state
        
                    # Add a new vehicle unless a collision occurs or ego reaches the end
                    if not done and not ego_reached_end:
                        for temp_idx in sorted(del_cars, reverse=True):
                            if random.uniform(0.0, 1.0) < add_car_prob:
                                added = state.add_car(earliest)
                                if added:
                                    cars_status.append(1)
                                    if self.file_config["save_sim_df"]:
                                        sim_df = sim_df._append({"Episode":(num_episodes*state_no+episode_no),
                                                                "Time_Step":counter,
                                                                "State":"N/A","Car_ID":state.cars_info.shape[1],
                                                                "Lane_ID":state.cars_info2[0,-1],
                                                                "Position_X":state.cars_info[0,-1],
                                                                "Velocity_X":state.cars_info[2,-1],
                                                                "Acceleration_X":state.cars_info[3,-1],
                                                                "Level_k":state.cars_info2[5,-1],
                                                                "Action":state.cars_info2[1,temp_idx],
                                                                "Dynamic_Action":state.cars_info2[5,temp_idx]},
                                                               ignore_index=True);
                    
                    # Save current ego frame to the ego dataframe
                    if self.file_config["save_ego_df"]:
                        if self.levelk_config["ego"] == 4:
                            if self.levelk_config["random_dynamic_strategy"]:                                                  
                                ego_df_row = [(num_episodes*state_no+episode_no), 
                                              counter-1, ego_state]+currentstate_list+\
                                    q_values1+q_values2+q_values3+\
                                        [dynamic_actionget, dynamic_action_type,
                                         actionget,state.cars_info[3,0],reward]
                            else:
                                ego_df_row = [(num_episodes*state_no+episode_no), 
                                              counter-1, ego_state]+currentstate_list+\
                                    dynamic_q_values+q_values1+q_values2+q_values3+\
                                        [dynamic_actionget, dynamic_action_type,
                                         actionget,state.cars_info[3,0],reward]

                                
                        elif 1 <= self.levelk_config["ego"] <= 3:
                            ego_df_row = [(num_episodes*state_no+episode_no), counter-1, 
                                              ego_state]+currentstate_list+q_values+[
                                                  actionget, action_type,
                                                  state.cars_info[3,0],reward]      
                        else:
                            ego_df_row = [(num_episodes*state_no+episode_no), counter-1, 
                                              ego_state]+currentstate_list+[state.cars_info2[1,0]]
                                           
                        ego_df = ego_df._append(
                            pd.DataFrame(list([ego_df_row]),
                                         columns=ego_df.columns),ignore_index=True)           
                    
                    # Save current frame to the simulation dataframe
                    if self.file_config["save_sim_df"]:            
                        with open(self.files["sim_df"],'a') as f:                
                            sim_df.to_csv(f, index=None, header=f.tell()==0)
                            del sim_df
                        
                    # Breaks the episode if ego cars crashes
                    if done:
                        #Write the state on which a collision occured on a file
                        col_df = pd.DataFrame(columns=COL_DF_COLS);
                        col = [(num_episodes*state_no+episode_no),numcars,counter
                               ]+currentstate_list+[state.cars_info2[1,0],ego_state,state.cars_info2[5,0]]
                        col_df = col_df._append(pd.DataFrame(
                            list([col]),columns=col_df.columns),ignore_index=True)
                        with open(self.files["col_df"],'a') as f:                
                            col_df.to_csv(f, index=None, header=f.tell()==0)
                            
                        # Ignore if a vehicle merges into ego
                        if ego_state == 1:
                            collision_type1_count += 1
                            print("==============================================")
                            print("================IGNORE CRASH==================")
                            print("==============================================\n")    
                            break
                         
                        print('Episode is forced to stop at step ' + str(counter) + ': '+ \
                              state.translate_car0_state(ego_state))
                        if self.levelk_config["ego"] == 4:
                            print('Dynamic Action Type: '+ dynamic_action_type +'\n' + 'Action Type: '+ action_type +'\n', end = '\n')
                        elif 1 <= self.levelk_config["ego"] <= 3:
                            print('Action Type: '+ action_type +'\n', end = '\n')
                     
                        collision_count += 1
                        break
                    
                    # Long episode takes more than a maximum number of steps
                    if not ego_reached_end and counter >= max_episode_length:
                        long_episodes += 1
                        print("Length of the episode exceeds max length " + \
                              str(max_episode_length) + '\n');
                        break
                
                if self.file_config["save_ego_df"]:            
                    with open(self.files["ego_df"],'a') as f:                
                        ego_df.to_csv(f, index=None, header=f.tell()==0)
                        del ego_df
                    
                # Ego reached the end without any collision                
                if not done and ego_reached_end:
                    print("========================================")
                    print("Car0 reached end of road at step " + str(counter))
                    if self.levelk_config["ego"] == 4:
                        print('Dynamic Action Type: '+ dynamic_action_type)
                        print('Action Type: '+ action_type)
                    elif 1 <= self.levelk_config["ego"] <= 3:
                        print('Action Type: '+ action_type)
                    print("========================================"+'\n', end = '\n')
                
                if num_episodes*state_no+episode_no-long_episodes-collision_type1_count + 1 == 0:
                    print('***Collision Percentage: 0.0%')
                else:
                    print('***Collision Percentage: ' + str(100 * collision_count / (
                        num_episodes*state_no+episode_no-long_episodes-collision_type1_count + 1)) + '%')
                
                print('***Long Episode Percentage: ' + str(100 * long_episodes / (
                    num_episodes*state_no+episode_no + 1)) + '%\n',end ='\n')
        
            state = None             

    def get_analyze_config(self):
        """
        Creates a list of configuration to be used by the SimulationAnalyzer object

        Returns
        -------
        analyze_config : dictionary
            List of configurations for SimulationAnalyzer.

        """
        analyze_config = {"files":{"sim_df_file": self.files["sim_df"],
                                    "ego_df_file": self.files["ego_df"],
                                    "col_df_file": self.files["col_df"],
                                    "analysis_df_file": self.analysis_files["analysis_df_file"],
                                    "ego_df_analysis_file": self.analysis_files["ego_df_analysis_file"]},
                           "pickle":{"long_eps_pickle": self.analysis_files["long_eps_pickle"],
                                     "succ_eps_pickle": self.analysis_files["succ_eps_pickle"],
                                     "crash_eps_pickle": self.analysis_files["crash_eps_pickle"]},
                           "num_episodes": self.simulation_config["num_episodes"],
                           "car_population":self.simulation_config["car_population"],
                           "ego":self.levelk_config["ego"]}
        
        return analyze_config
    
def main():
    path = os.path.dirname(os.getcwd()) + "/experiments/some_title/"
    
    ego_level = 4 # Level-k Policy for ego (0: Level-0, 1: Level-1, 2: Level-2, 3: Level-3, 4: Dynamic)
    models = {1: 99, # Level-1 Model
              2: 99, # Level-2 Model
              3: 99, # Level-3 Model
              4: 99} # Dynamic Model
    
    ego_models = [99] # Ego models to be iterated during the simulation
    # -2: Rnd Single, -1: Rnd Mixed, 0: Level-0, 1: Level-1, 2: Level-2, 3: Level-3, 4: Dynamic
    vs = [0,1] # Environment type list to be iterated during the simulation
    
    num_episodes = 500 # Number of episodes for each population group
    car_population = [12,16,20,24] # List used for changing population
    
    for ego_model in ego_models:
        for vs_temp in vs:
            # Level-k Configuration Parameters
            distribution_bound = 1.0 # Bound for uniform distribution to select the ratio of different levels
            boltzmann_sampling = True # If true, agents select action via Boltzmann sampling
            dynamic_driving_boltzmann_sampling = True # Driving actions are sampled via Boltzmann for dynamic agent
            random_dynamic_strategy = False # Dynamic agent selects strategies randomly 

            # Simulation Configuration Parameters           

            ego_lane = -1 # -1: Random / 0: Lane-0 (Ramp) / 1: Lane-1 (Main Road)
            add_car_prob = 0.7 # Add a new car with this probability when enough space is available in the beginning of the road
            bias_lane0vs1 = 0.7 # Add a new car on lane-1 with this probability
            maxcars_onlane0 = 7 # Maximum number of cars on the ramp
            max_episode_length = 300/Params.timestep # Max number of steps to stop an episode
    
            # File Configuration Parameters
            save_sim_df = True # Save simulation into csv file
            save_ego_df = False # Save ego information into csv file
            
            if ego_level != 4:
                directory = "level"+str(ego_level)+"/simulation/"
            else:
                directory = "dynamic/simulation/"
                
            sim_models = {1: models[1], 2: models[2], 3: models[3], 4: models[4]}
            if 1 <= vs_temp <= 3:
                sim_models[vs_temp] = models[vs_temp]
            elif -2 <= vs_temp <= -1 or vs_temp == 4:
                sim_models[1] = models[1]
                sim_models[2] = models[2]
                sim_models[3] = models[3]
                
            sim_models[ego_level] = ego_model
            if ego_level == 4:
                sim_models[1] = models[1]
                sim_models[2] = models[2]
                sim_models[3] = models[3]       
            
            # Level-k Configuration Dictionary for Simulation
            levelk_config = {"ego": ego_level,
                              "vs": 1,
                              "models":{1: sim_models[1],
                                        2: sim_models[2],
                                        3: sim_models[3],
                                        4: sim_models[4]},
                              "boltzmann_sampling": boltzmann_sampling,
                              "dynamic_driving_boltzmann_sampling": dynamic_driving_boltzmann_sampling,
                              "random_dynamic_strategy": random_dynamic_strategy,
                              "distribution_bound":distribution_bound}
            
            # Simulation Configuration Dictionary for Simulation
            simulation_config = {"num_episodes": num_episodes,
                                  "car_population": car_population,
                                  "ego_lane": ego_lane,
                                  "add_car_prob": add_car_prob,
                                  "bias_lane0vs1": bias_lane0vs1,
                                  "maxcars_onlane0": maxcars_onlane0,
                                  "max_episode_length": max_episode_length}
            
            # File Configuration Dictionary for Simulation
            file_config = {"path": path,
                            "directory": directory,
                            "save_sim_df": save_sim_df,
                            "save_ego_df": save_ego_df}
            
            sim = Simulation(levelk_config = levelk_config,
                              simulation_config = simulation_config,
                              file_config = file_config)
            
            sim.run()
            
            analyzer = SimulationAnalyzer(analyze_config = sim.get_analyze_config()) 
            analyzer.analyze()
    
if __name__ == '__main__':
    main()