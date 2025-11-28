#This file implements the training of level-k/dynamic agents in highway merging
# © 2020 Cevahir Köprülü All Rights Reserved

from Params import Params
from src.Message import Message
from src.DQNAgent import DQNAgent
from src.State import State
from src.DynamicDQNAgent import DynamicDQNAgent
from src.Simulation import Simulation
from src.SimulationAnalyzer import SimulationAnalyzer
import numpy as np
import os 
import pandas as pd
import random
import math
from pathlib import Path
from collections import deque
import logging 
# logging.getLogger('tensorflow').setLevel(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} ) 
# sess = tf.Session(config=config) 
# keras.backend.set_session(sess)

TRAINING_DF_COLS = ['Episode', 'Time_Step', 'State', 'Car_ID', 'Lane_ID',
                    'Position_X', 'Velocity_X', 'Level_k', 'Action', 'Dynamic_Action']

EGO_DF_DYN_COLS = ['Episode', 'Time_Step', 'State', 'fs_d', 'fs_v', 'fc_d', 
                   'fc_v', 'rs_d', 'rs_v', 'velocity', 'lane', 'dist_end_merging', 
                   'q-lev1', 'q-lev2', 'q-lev3', 'q-maintain1', 'q-accel1', 
                   'q-decel1', 'q-hard_accel1', 'q-hard_decel1', 'q-merge1',
                   'q-maintain2', 'q-accel2', 'q-decel2', 'q-hard_accel2', 
                   'q-hard_decel2', 'q-merge2', 'q-maintain3', 'q-accel3', 
                   'q-decel3', 'q-hard_accel3', 'q-hard_decel3', 'q-merge3', 
                   'Dynamic_Action', 'Dynamic_Action_Type', 'Action']

EGO_DF_LEVK_COLS = ['Episode', 'Time_Step', 'State', 'fs_d', 'fs_v', 'fc_d', 'fc_v', 
               'rs_d', 'rs_v', 'velocity', 'lane', 'dist_end_merging', 'q-maintain', 
               'q-accel', 'q-decel', 'q-hard_accel', 'q-hard_decel', 'q-merge', 
               'Action', 'Action_Type']

CRASH_HIST_DF_DYN_COLS = ['Episode','numcars', 'Time_Step', 'fs_d', 'fs_v',
                          'fc_d', 'fc_v', 'rs_d', 'rs_v', 'velocity', 'lane', 
                          'dist_end_merging', 'q-lev1', 'q-lev2', 'q-lev3',
                          'Dynamic Action', 'Dynamic Action Type', 'Action',
                          'Action Type', 'Reward', 'Crash Type']

CRASH_HIST_DF_COLS = ['Episode', 'numcars', 'Time_Step', 'fs_d', 'fs_v', 'fc_d', 
                      'fc_v', 'rs_d', 'rs_v', 'velocity', 'lane', 
                      'dist_end_merging', 'q-maintain', 'q-accel', 'q-decel', 
                      'q-hard_accel', 'q-hard_decel', 'q-merge', 'Action', 
                      'Action Type', 'Reward','  Crash Type']

class Training():
    
    def __init__(self, levelk_config, training_config, file_config):
        self.levelk_config = levelk_config
        self.file_config = file_config
        self.training_config = training_config
        self.state_size = Params.num_observations
        self.action_size = Params.num_actions
        self.dynamic_state_size = Params.num_observations * \
            Params.dynamic_history + Params.dynamic_history_action * \
                (Params.dynamic_history-1)
        self.dynamic_action_size = Params.num_dynamic_actions  # num of levels trained     
        self.training_agent = self._create_training_agent()
        self.models = self._load_models()     
        self.files = self._create_files()
        
    def _create_training_agent(self):
        """
        Create the training agent according to its level-k setting

        Returns
        -------
        training_agent : DQNAgent or DynamicDQNAgent Object

        """
        if 1 <= self.levelk_config["ego"] <= 3:
            training_agent = DQNAgent(self.state_size, self.action_size) # Level-k Agent
            print("Training Agent: Level "+str(self.levelk_config["ego"]))
        else:
            agent_levelk_paths = {1:None,2:None,3:None}
            for i in range(1,4):
                if self.levelk_config["models"][i] != None: #was loaded according to the models of levels if entered any
                    agent_levelk_paths[i] = self.file_config["path"]+"/level"+str(i)+\
                                              "/training/models/model"+\
                                              str(self.levelk_config["models"][i])
            levelk_config = {"paths": agent_levelk_paths, "boltzmann_sampling": self.levelk_config["dynamic_driving_boltzmann_sampling"]}
            training_agent = DynamicDQNAgent(self.dynamic_state_size,
                                             self.dynamic_action_size, 
                                             self.state_size, self.action_size, 
                                             levelk_config) 
            print("Training Agent: Dynamic")

        return training_agent
        
    def _load_models(self):
        """
        Load required models to be used for the surrounding agents

        Returns
        -------
        models : dictionary
            Keeps DQNAgent and DynamicDQNAgent objects.

        """
        models = {1:None,2:None,3:None,4:None}
        agent_levelk_paths = {1:None,2:None,3:None}
        for i in range(1,4):
            if self.levelk_config["models"][i] != None:
                agent_levelk_paths[i] = self.file_config["path"]+"/level"+str(i)+\
                                          "/training/models/model"+\
                                          str(self.levelk_config["models"][i])
                
        if self.levelk_config["vs"] <= -1:  #check for dynamic/ random dynamic
            for i in range(1,4):
                agentlevk = DQNAgent(self.state_size, self.action_size) # Level-k Agent
                if (agent_levelk_paths[i] != None):
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
            dynamic_agent = DynamicDQNAgent(self.dynamic_state_size,self.dynamic_action_size, 
                                            self.state_size, self.action_size, 
                                            levelk_config) 
            if not self.levelk_config["random_dynamic_strategy"]:
                dynamic_agent.load(self.file_config["path"]+\
                                   "/dynamic/training/models/model"+\
                                       str(self.levelk_config["models"][4]))
            dynamic_agent.T = dynamic_agent.MIN_T 
            dynamic_agent.boltzmann_sampling = self.levelk_config["boltzmann_sampling"]
            dynamic_agent.random_dynamic_strategy = self.levelk_config["random_dynamic_strategy"]
            models[4] = dynamic_agent    
            
        return models
       
    def _create_files(self):
        """
        Creates directories, assigns file names

        Returns
        -------
        files : dictionary
            Dictionary to keep necessary file names for recording.

        """
        path = self.file_config["path"]
        directory = self.file_config["directory"]
        Path("./"+path+directory+"/training_data").mkdir(parents=True, exist_ok=True)
        Path("./"+path+directory+"/agent_data").mkdir(parents=True, exist_ok=True)
        Path("./"+path+directory+"/target_weights").mkdir(parents=True, exist_ok=True)
        Path("./"+path+directory+"/models").mkdir(parents=True, exist_ok=True)
        
        files = {"training_df":None, 
                 "crash_history_df":None, 
                 "ego_df":None,
                 "reward": None,
                 "collision": None,
                 "train_hist":None}
        

        #this line can be computed in parallel


        #os.remove(files["crash_history_df"])      
        
        if not self.training_config["retraining_config"]["retrain"]:
            generic_file = path + directory + "training_data"
            
            if self.file_config["save_crash_history_df"]:
                files["crash_history_df"] = generic_file  + "/crash_history.csv" # assigns the path
                if os.path.exists(files["crash_history_df"]):
                    os.remove(files["crash_history_df"])      
                
            if self.file_config["save_training_df"]:
                files["training_df"] = generic_file + "/training.csv"
                if os.path.exists(files["training_df"]):
                    os.remove(files["training_df"])             
                
            if self.file_config["save_ego_df"]:
                files["ego_df"] = generic_file + "/ego.csv"
                if os.path.exists(files["ego_df"]):
                   os.remove(files["ego_df"]) 
    
            files["reward"] = generic_file + "/reward.dat"
            if os.path.exists(files["reward"]):
                os.remove(files["reward"])
                
            files["collision"] = generic_file + "/collision.dat"         
            if os.path.exists(files["collision"]):
                os.remove(files["collision"])     

            files["train_hist"] = generic_file + "/train_hist.dat"         
            if os.path.exists(files["train_hist"]):
                os.remove(files["train_hist"])  

        return files
    
    def _prepare_retraining(self):
        """
        Prepares the Training object to start retraining from a given state
        Loads training agent properties: Model/Target Model/Memory/Last Step Number
        Assigns new file names

        Returns
        -------
        total_timesteps : int
            The previous timestep from which the training will restart.

        """
        path = self.file_config["path"]
        directory = self.file_config["directory"]
        
        first_state_reset = self.training_config["retraining_config"]["first_state_reset"]
        self.training_agent.load(path + directory + \
                                 "/models/model" + str(first_state_reset-1),
                                 path+directory + "/target_weights/target_weight" + \
                                     str(first_state_reset-1) + ".h5",retrain=True)
        self.training_agent.load_memory(path + directory + "agent_data/agent_memory" + \
                                        str(first_state_reset-1) + ".pickle")
        total_timesteps = self.training_agent.load_config(path + directory + "" + \
                                                          "/agent_data/agent_config" + \
                                                              str(first_state_reset-1) + \
                                                                  ".pickle")
        
        generic_file = path + directory + "training_data"

        self.files["reward"] = generic_file + "/retrain_" + \
            str(first_state_reset) + "_reward.dat"
        if os.path.exists(self.files["reward"]):
            os.remove(self.files["reward"])  
            
        self.files["collision"] = generic_file + "/retrain_" + \
            str(first_state_reset) + "_collision.dat"
        if os.path.exists(self.files["collision"]):
            os.remove(self.files["collision"])  
            
        self.files["train_hist"] = generic_file + "/retrain_" + \
            str(first_state_reset) + "_train_hist.dat"
        if os.path.exists(self.files["train_hist"]):
            os.remove(self.files["train_hist"])              
            
        if self.file_config["save_training_df"]:
            self.files["training_df"] = generic_file + "/retrain_" + \
                str(first_state_reset) + "_training.csv";  
            if os.path.exists(self.files["training_df"]):
                os.remove(self.files["training_df"]) 
                
        if self.file_config["save_ego_df"]:
            self.files["ego_df"] = generic_file + "/retrain_" + \
                str(first_state_reset) + "_ego.csv";     
            if os.path.exists(self.files["ego_df"]):
                os.remove(self.files["ego_df"]) 
        
        if self.file_config["save_crash_history_df"]:
            self.files["crash_history_df"] = generic_file + "/retrain_" + \
                str(first_state_reset) + "_crash_history.csv";     
            if os.path.exists(self.files["crash_history_df"]):
                os.remove(self.files["crash_history_df"])

                
        return total_timesteps
    
    def record_training_history(self, episode, timestep, loss):
        """
        Records info about Q-loss, average weight/bias of every layer

        Parameters
        ----------
        episode : int
            No of the episode to be recorded.
        timestep : int
            No of the timestep to be recorded.
        loss : flaot
            Q-loss to be recorded.

        Returns
        -------
        None.

        """
        ave_weight, ave_bias = self.training_agent.get_average_layer_weights()
        train_hist_line = str(episode)+"\t"+\
            str(timestep)+"\t"+str(loss)
       # file = open(self.files["train_hist"], 'w') #ask if better
        
        for idx_ave in range(len(ave_weight)):
            train_hist_line += "\t"+str(ave_weight[idx_ave])+"\t"+str(ave_bias[idx_ave])
        file = open(self.files["train_hist"], 'a')
        file.write(train_hist_line+"\n")
        file.close()

    def record_reward(self, episode, timestep, reward):
        """
        Records reward information

        Parameters
        ----------
        episode : int
            No of the episode to be recorded.
        timestep : int
            No of the timestep to be recorded.
        reward : flaot
            Reward to be recorded.

        Returns
        -------
        None.

        """
        file = open(self.files["reward"], 'a')
        file.write('' + str(episode) + '\t' + \
                   str(timestep) + '\t' + str(reward) + '\n')
        file.close()
        
    def record_collision(self, episode, ego_final_state):
        """
        Records collision information

        Parameters
        ----------
        episode : int
            No of the episode to be recorded.
        ego_final_state : int
            Final state of the ego.

        Returns
        -------
        None.

        """
        # -1: Moving
        # 1: Another vehicle merged into ego
        # 4: Ego reached the end of the road
        is_collision = int((ego_final_state not in [-1, 1, 4, 6, 7]))
        
        file = open(self.files["collision"], 'a')
        file.write('' + str(episode) + '\t' + \
                   str(is_collision) + '\t' + str(ego_final_state) +'\n')
        file.close()  
    
    def remember_frame(self, currentstate, actionget, nextstate, reward, done, 
                       ego_reached_end, state_size, dynamic_actionget = None):
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
        nextstate : numpy array
            Next state of the ego.
        reward : float
            Reward taken with the current action.
        done : bool
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
        if self.levelk_config["ego"] == 4:
            temp_nextstate = nextstate.copy()
            nextstate = currentstate.copy()
            if Params.dynamic_history_action:
                scale_action = 1.0 if not Params.scale_action else (Params.num_actions-1.0)
                nextstate_concat = np.concatenate((nextstate[0,0,:self.state_size].copy(),
                                                   [actionget/scale_action],
                                                   nextstate[0,0,self.state_size:-state_size].copy()))
                nextstate[0,0,self.state_size:] = nextstate_concat
            else:
                nextstate[0,0,self.state_size:] = nextstate[0,0,:-self.state_size].copy()
            nextstate[:,:, :self.state_size] = temp_nextstate.copy()
            self.training_agent.remember(currentstate, 
                                         dynamic_actionget, 
                                         reward, nextstate, 
                                         (done or ego_reached_end))
        else:
            self.training_agent.remember(currentstate, actionget, 
                                         reward, nextstate, 
                                         (done or ego_reached_end))
        
    def randomize_levelk_distribution(self, state_no):
        """
        Randomly sample the ratio of level-k vehicles in an episode

        Returns
        -------
        levelk_config : dictionary
            Includes ego and other vehicle level-k setting.

        """
        levelk_config = {"ego": self.levelk_config["ego"],
                         "others": None}
        
        if self.levelk_config["vs"] == -1:
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
            #print("Distribution: L0="+str(round(l0,2))+" L1=" + str(round(l1,2))+
                 # " L2=" + str(round(l2,2))+" L3=" + str(round(l3,2)))
            
            levelk_config["others"] = np.array([l0, l1, l2, l3, 0.0])  #probabilities there 
        elif self.levelk_config["vs"] == -2:  #randomly chooses an environment 
            #env_levk = random.randint(0,3)
            if state_no < 20:
                env_levk = 0
            elif state_no < 40:
                env_levk = 1
            elif state_no < 60:
                env_levk = 2
            elif state_no < 80:
                env_levk = 3
            else:
                env_levk = random.randint(0,3)

            
            levelk_config["others"] = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            levelk_config["others"][env_levk] = 1.0
            print("Level-"+str(env_levk)+" Environment")
            
        return levelk_config  
      
    def check_ego_df_condition(self,currentstate):
        """
        Check currentstate to see if ego is in a state to be recorded

        Parameters
        ----------
        currentstate : lsit
            Includes the current state of the ego.

        Returns
        -------
        bool
            True to state that current state should be recorded.

        """
        # record = False
        
        # ##########################################################
        # # Stopping case on the ramp, used in the reward function #
        # ##########################################################
        # vel = currentstate[6]*Params.max_speed
        # if vel <= -Params.hard_decel_rate*Params.timestep:
        #     dist_end_merging = currentstate[-1]*Params.merging_region_length
        #     fc_d = currentstate[2]*Params.max_sight_distance
            
        #     # If the headway distance is greater than $far_distance$ while
        #     # commuting at a slow speed, record this state
        #     if fc_d >= Params.far_distance:
        #         record = True
        #     # If the ego is on the ramp and close to the end of the merging region
        #     # less than $far_distance$, proceed
        #     elif currentstate[7] == 0 and 0 <= dist_end_merging <= Params.far_distance:
        #         fl_d = currentstate[0]*Params.max_sight_distance
        #         rl_d = currentstate[4]*Params.max_sight_distance
        #         # If there is enough space on the main road, record this state
        #         if fl_d >= Params.close_distance and rl_d <= -1.5*Params.far_distance:
        #         # if fl_d >= Params.nominal_distance and rl_d <= -1.5*Params.far_distance:
        #             record = True
        record = True
        return record
    
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
        states = {1: {0:[], 1:[]},   #keys: 1,2,3,4 -> levels. In values: merging/not merging : keys, states are the inner values?
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
        for idx_vehicle in range(1,state.cars_info.shape[1]):#among the cars in the cars_info
            levk = state.cars_info2[5,idx_vehicle] # Level-k strategy
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
 
                cars[levk][int(merging)].append(idx_vehicle) #in ith level of car, append the instance of the car in cars_info of leveli
                if 1 <= levk <= 3:
                    states[levk][int(merging)].append(currentstate_temp)
                else:
                    state_temp = np.zeros((1,self.dynamic_state_size))
                    state_temp[0,0:self.state_size] = currentstate_temp[0,:].copy()
                    
                    # If $dynamic_history$ is more than 1, stack the current frame
                    for t_i in range(1,Params.dynamic_history):
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
                                                      
        acts = np.zeros((state.cars_info.shape[1])) #initializes "acts" array of size the number of cars in cars_info
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
            acts[np.asarray(cars[4][0])], dyn_acts[np.asarray(cars[4][0])] = self.models[4].act_inbatch(
                np.asarray(states[4][0]),remove_merging=True)    
   
        # Update dyn_msg dictionary
        for t_i in range(Params.dynamic_history-1,0,-1):
            if len(dyn_msg[t_i]) != 0:
                if t_i == 1:
                    for i,dyn_id in enumerate(cars[4][0]):
                        if Params.dynamic_history_action:
                            scale_action = 1.0 if not Params.scale_action else (Params.num_actions-1.0)
                            dyn_msg[t_i][dyn_id] = np.concatenate((states[4][0][i][0,0:self.state_size].copy(),
                                                                   [acts[dyn_id]/scale_action]))
                        else:
                            dyn_msg[t_i][dyn_id] = states[4][0][i][0,0:self.state_size].copy()
                    for i,dyn_id in enumerate(cars[4][1]):
                        if Params.dynamic_history_action:
                            scale_action = 1.0 if not Params.scale_action else (Params.num_actions-1.0)
                            dyn_msg[t_i][dyn_id] = np.concatenate((states[4][1][i][0,0:self.state_size].copy(),
                                                                   [acts[dyn_id]/scale_action]))
                        else:
                            dyn_msg[t_i][dyn_id] = states[4][1][i][0,0:self.state_size].copy()
                else:
                    dyn_msg[t_i] = dyn_msg[t_i-1].copy()
                
        return cars[0], acts, dyn_acts, dyn_msg
    
    def run(self):
        """
        Run the training

        Returns
        -------
        None.

        """
        ego_lane = self.training_config["ego_lane"]
        add_car_prob = self.training_config["add_car_prob"]
        bias_lane0vs1 = self.training_config["bias_lane0vs1"]
        maxcars_onlane0 = self.training_config["maxcars_onlane0"]
        car_population = self.training_config["car_population"]
        ignore_stopping = self.training_config["ignore_stopping"]
        random_car_population = self.training_config["random_car_population"]
        change_car_population_every_ep = self.training_config["change_car_population_every_ep"]
        curriculum_config = self.training_config["curriculum_config"]
        
        num_state_resets = self.training_config["num_state_resets"] 
        num_episodes = self.training_config["num_episodes"]
        first_phase = self.training_config["first_phase"]
        target_up = self.training_config["target_up"]
        batch_size = self.training_config["batch_size"]
        replay_start_size = self.training_config["replay_start_size"]
        stop_sinusoidal_state = self.training_config["stop_sinusoidal_state"]
        max_episode_length = self.training_config["max_episode_length"]
        boltzmann_decay_end_state = self.training_config["boltzmann_decay_end_state"]
        
        # Level-k agent distribution in the 'environment'
        # 0: Level-0
        # 1: Level-1
        # 2: Level-2
        # 3: Level-3
        # 4: Dynamic
        distribution = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        distribution[self.levelk_config["vs"]] = 1.0  #only the prev level loaded as 1 (one-hot)
        levelk_config = {"ego": self.levelk_config["ego"],
                         "others": distribution}
        
        if self.file_config["save_crash_history_df"]:
            crash_history_size = self.file_config["crash_history_size"]
            crash_history_first_ep = self.file_config["crash_history_first_ep"]  
    
         
        if self.file_config["save_ego_df"]:
            if self.levelk_config["ego"] == 4:
                ego_df_columns = EGO_DF_DYN_COLS
            else:
                ego_df_columns = EGO_DF_LEVK_COLS
            

        if self.file_config["save_crash_history_df"]:
            if self.levelk_config["ego"] == 4:
                crash_history_df_columns = CRASH_HIST_DF_DYN_COLS
            else:
                crash_history_df_columns = CRASH_HIST_DF_COLS
        
        total_timesteps = 0 # Timestep passed since the start of the training
        first_state_reset = 0
        done = False # True if ego is in a collision
        
        # Reload last settings for retraining
        if self.training_config["retraining_config"]["retrain"]:
            total_timesteps = self._prepare_retraining()
            first_state_reset = self.training_config["retraining_config"]["first_state_reset"]
        
        if total_timesteps == 0:
            self.record_training_history(episode = 0, timestep = 0, loss = 0)
            
        state_no = first_state_reset
        while state_no < num_state_resets: #model
        # for state_no in range(first_state_reset, num_state_resets):
            
            # Change to the random car population phase when sinusoidal phase ends
            if state_no >= stop_sinusoidal_state:
                random_car_population = True 

            episode_no = 0
            while episode_no < num_episodes:
               
                print('Episode ' + str(num_episodes*state_no+episode_no))  # episode 100*model_no+ episode(<100)
                
                # $numcars$ selection
                if curriculum_config["enable_curriculum"] and state_no < curriculum_config["curriculum_end_state"]:
                    print("Curriculum numcars selection")
                    curriculum_current_end = state_no//curriculum_config["curriculum_skip_every_state"]
                    # numcars = int(random.sample(car_population[0:curriculum_current_end+1],1)[0])
                    numcars = car_population[curriculum_current_end]
                else:
                    if state_no < first_phase: #for first 2 models- 2:first phase
                        numcars = car_population[0]  #choose the smallest environment for 2 models (200 epsiodes)
                    else:
                        if random_car_population:  #after sinusoidal stops : 50th model
                            print("Random numcars selection")
                            numcars = int(random.sample(car_population[0:(
                                len(car_population)//2+2)],1)[0])   #makes the randomization, repetitions eliminated
                        else:
                            iter_car_population = (((state_no*num_episodes+episode_no)-\
                                                    first_phase*num_episodes) // \
                                                   change_car_population_every_ep) % \
                                len(car_population)      #correctly cycles the population list
                            numcars = car_population[iter_car_population]  
                print('Training with ' + str(numcars) + " cars")

                long_episode_flag = False
                ego_reached_end = False
                counter = 0 # Steps taken in an episode
                
                # Obtain a randomly sampled level-k distribution to set the 
                # environment
                if self.levelk_config["vs"] == -1:
                    levelk_config = self.randomize_levelk_distribution(state_no)
                elif self.levelk_config["vs"] == -2:
                    if (state_no*num_episodes+episode_no) % self.training_config["rnd_levk_env_eps"] == 0:  #in each rnd_levk_env_eps episodes
                        levelk_config = self.randomize_levelk_distribution(state_no)
                # Initializes the environment
                state = State(numcars, ego_lane, bias_lane0vs1, maxcars_onlane0, 
                              levelk_config)
                numcars = state.cars_info.shape[1]
                
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
                else:
                    state_size = self.state_size
                    
                ego_reached_end = False
                reward = 0 
                counter = 0 # Current step in the episode
                ego_state = -1 
                actionget = 0

                # Initialize crash history dataframe and queue
                if (self.file_config["save_crash_history_df"] and 
                    (episode_no + state_no * num_episodes) >= crash_history_first_ep):
                    
                    crash_history_df = pd.DataFrame(columns=crash_history_df_columns)
                    crash_history = deque(maxlen=crash_history_size) 

                # Initialize training dataframe and append the initial frame
                if self.file_config["save_training_df"]:
                    training_df = pd.DataFrame(columns= TRAINING_DF_COLS);
                    
                    for temp_idx in range(state.cars_info.shape[1]):
                        if temp_idx == 0:
                            training_df = training_df._append({"Episode":(num_episodes*state_no+episode_no),
                                                    "Time_Step":counter,
                                                    "State":ego_state,"Car_ID":temp_idx,
                                                    "Lane_ID":state.cars_info2[0,temp_idx],
                                                    "Position_X":state.cars_info[0,temp_idx],
                                                    "Velocity_X":state.cars_info[2,temp_idx],
                                                    "Level_k":state.cars_info2[5,temp_idx], 
                                                    "Action": state.cars_info2[1,temp_idx], 
                                                    "Dynamic_Action":state.cars_info2[5,temp_idx]},ignore_index=True);
                        else:
                            training_df = training_df._append({"Episode":(num_episodes*state_no+episode_no),
                                                    "Time_Step":counter,
                                                    "State":"N/A","Car_ID":temp_idx,
                                                    "Lane_ID":state.cars_info2[0,temp_idx],
                                                    "Position_X":state.cars_info[0,temp_idx],
                                                    "Velocity_X":state.cars_info[2,temp_idx],
                                                    "Level_k":state.cars_info2[5,temp_idx],  
                                                    "Action": state.cars_info2[1,temp_idx], 
                                                    "Dynamic_Action":state.cars_info2[5,temp_idx]},ignore_index=True);
                     
                    with open(self.files["training_df"],'a') as f:                
                        training_df.to_csv(f, index=None, header=f.tell()==0)
                        del training_df
                
                # Initiliaze ego dataframe
                if self.file_config["save_ego_df"]:
                   ego_df = pd.DataFrame(columns=ego_df_columns);             
                
                while not ego_reached_end:
                    # Earlist list is used to add a new vehicle
                    # [Position on the ramp, Speed on the ramp, 
                    # earliest_vehicle_pos_main_road, earliest_vehicle_speed_main_road]
                    earliest = [Params.init_size,Params.max_speed,Params.init_size,Params.max_speed]
                    
                    # Initialize training dataframe
                    if self.file_config["save_training_df"]:
                        training_df = pd.DataFrame(columns = TRAINING_DF_COLS);
              
                    # State/Observation of the ego car
                    currentstate_list = state.get_Message(0,normalize=True) 
                    currentstate = np.reshape(currentstate_list, [1, 1, self.state_size])
                    
                    # Dynamic ego section
                    if self.levelk_config["ego"] == 4:
                        # Use previous ego actions for the state of the dynamic agent
                        if Params.dynamic_history_action:
                            scale_action = 1.0 if not Params.scale_action else (Params.num_actions-1.0)
                            temp_concat =  np.concatenate((ego_dyn_state[0,0,:self.state_size].copy(),
                                                        [actionget/scale_action],
                                                        ego_dyn_state[0,0,self.state_size:-state_size].copy())) 
                            ego_dyn_state[0,0,self.state_size:] = temp_concat
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
                    if (((Params.start_merging_point + Params.carlength) < state.cars_info[0,0] <
                         Params.end_merging_point) and state.cars_info2[0,0] == 0):
                        remove_merging = False
                    else:
                        remove_merging = True
                        
                    # Get ego related info if ego or crash history dataframes 
                    # are recorded
                    if ((self.file_config["save_ego_df"] and 
                         self.check_ego_df_condition(currentstate_list)) or 
                        self.file_config["save_crash_history_df"]):
                        
                        ego_info = self.training_agent.act(state=currentstate, 
                                                           remove_merging=remove_merging, 
                                                           get_qvals=True) #action taken by the ego car
                        if self.levelk_config["ego"] == 4:
                            # Dynamic Model Output
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
                        ego_info = self.training_agent.act(state=currentstate, 
                                                           remove_merging=remove_merging, 
                                                           get_qvals=False) #action taken by the ego car
                        if self.levelk_config["ego"] == 4:
                            dynamic_action_type = ego_info[0][0]
                            dynamic_actionget = ego_info[0][1]
                            action_type = ego_info[1][0]
                            actionget = ego_info[1][1]
                        elif 1 <= self.levelk_config["ego"] <= 3:
                            action_type = ego_info[0]
                            actionget = ego_info[1]
                                
                    cars_lev0, acts, dyn_acts, dyn_msg = self.get_state_action(state, dyn_msg)

                    counter += 1
                    total_timesteps += 1
                    del_cars = []
                    cars_lev0_counter = 0
                    for temp_idx in range(state.cars_info.shape[1]):
                        if temp_idx == 0:
                            # Update the position of the ego car based on the selected action
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
                                    
                        if self.file_config["save_training_df"]:    
                            training_df = training_df._append({"Episode":(num_episodes*state_no+episode_no),
                                                    "Time_Step":counter,
                                                    "State": ego_state,
                                                    "Car_ID": temp_idx,
                                                    "Lane_ID":state.cars_info2[0,temp_idx],
                                                    "Position_X":state.cars_info[0,temp_idx],
                                                    "Velocity_X":state.cars_info[2,temp_idx],
                                                    "Level_k":state.cars_info2[5,temp_idx], 
                                                    "Action":state.cars_info2[1,temp_idx],
                                                    "Dynamic_Action":dyn_acts[temp_idx]+1},
                                                             ignore_index=True);
                            
                        # Update earlist list to add a new vehicle appropriately
                        if state.cars_info[0,temp_idx] < earliest[0 + 2*state.cars_info2[0,temp_idx]]:
                            earliest[0 + 2*state.cars_info2[0,temp_idx]] = state.cars_info[0,temp_idx]
                            earliest[1 + 2*state.cars_info2[0,temp_idx]] = state.cars_info[2,temp_idx]
                            
                    
                    state.check_reset_positions() # Reset vehicle positions if a collision occurs
                    nextstate = state.get_Message(0,normalize=True) # State reached by the ego car
                    [done, ego_state] = state.check_ego_state(nextstate,ignore_stopping=ignore_stopping) # Checks if there is a collision or not
                    reward = state.get_reward(done,nextstate) # Current reward
                    
                    # Change ego state as "Ego Reached the End" if it is so
                    if not done and ego_reached_end:
                        ego_state = 4
                 
                    # Add the last frame to the crash history queue
                    if (self.file_config["save_crash_history_df"] and 
                        (episode_no + state_no * num_episodes) >= crash_history_first_ep):
                        if state.cars_info2[5,0] == 4:
                            hist_one_step = [(num_episodes*state_no+episode_no), 
                                             numcars, counter]+\
                                currentstate_list+dynamic_q_values+[dynamic_actionget, 
                                                                    dynamic_action_type, 
                                                                    actionget, 
                                                                    action_type, 
                                                                    reward, 
                                                                    ego_state]
                        else:
                            hist_one_step = [(num_episodes*state_no+episode_no), 
                                             numcars, counter]+currentstate_list+\
                                q_values+[actionget, action_type, reward, ego_state] 
                                
                        crash_history.append(hist_one_step) 
                    
                    # Change the ego state in the training dataframe
                    if self.file_config["save_training_df"]:
                        training_df.iloc[-state.cars_info.shape[1],2] = ego_state
                    
                    # Add a new vehicle to the environment unless a collision 
                    # occurs and ego reaches the end
                    if not done and not ego_reached_end:
                        for temp_idx in sorted(del_cars, reverse=True):
                            if random.uniform(0.0, 1.0) < add_car_prob:
                                added = state.add_car(earliest)
                                if added:
                                    cars_status.append(1)
                                    if self.file_config["save_training_df"]:
                                        training_df = training_df._append({"Episode":(num_episodes*state_no+episode_no),
                                                                "Time_Step":counter,
                                                                "State":"N/A","Car_ID":state.shape[1],
                                                                "Lane_ID":state.cars_info2[0,-1],
                                                                "Position_X":state.cars_info[0,-1],
                                                                "Velocity_X":state.cars_info[2,-1],
                                                                "Level_k":state.cars_info2[5,-1],
                                                                "Action":state.cars_info2[1,temp_idx],
                                                                "Dynamic_Action":state.cars_info2[5,temp_idx]},
                                                                         ignore_index=True);
                    
                    # Ignore if the ego state is "A Car Merged into Ego"
                    if ego_state != 1:
                        nextstate = np.reshape(nextstate, [1, 1,self.state_size])                    
                        
                        # Append the current info about the ego to the ego dataframe
                        if self.file_config["save_ego_df"] and self.check_ego_df_condition(currentstate_list):
                            if self.levelk_config["ego"] == 4:
                                ego_df_row = [(num_episodes*state_no+episode_no), 
                                              counter-1, ego_state]+currentstate_list+\
                                    dynamic_q_values+q_values1+q_values2+q_values3+\
                                        [dynamic_actionget, dynamic_action_type,
                                         actionget]
                            else:
                                ego_df_row = [(num_episodes*state_no+episode_no), counter-1, 
                                                  ego_state]+currentstate_list+q_values+[
                                                      actionget, action_type]      

                            ego_df = ego_df._append(
                                pd.DataFrame(list([ego_df_row]),
                                             columns=ego_df.columns),ignore_index=True)  

                        # Append the last frame to the training dataframe
                        if self.file_config["save_training_df"]:            
                            with open(self.files["training_df"],'a') as f:                
                                training_df.to_csv(f, index=None, header=f.tell()==0)
                                del training_df                        
                        
                        # Remember the current transition
                        if self.levelk_config["ego"] == 4:
                            dynamic_action = dynamic_actionget
                        else:
                            dynamic_action = None
                        
                        self.remember_frame(currentstate = currentstate, 
                                            actionget = actionget,
                                            nextstate = nextstate,
                                            reward = reward,
                                            done = done,
                                            ego_reached_end = ego_reached_end,
                                            state_size = state_size,
                                            dynamic_actionget = dynamic_action)

                        # Experience replay, returned value is the MSE between
                        # the target model output and the main moedl output
                        current_training_loss = 0
                        if total_timesteps > replay_start_size and total_timesteps % 4 == 0:
                            current_training_loss = self.training_agent.replay(batch_size)[0] #Experience replay
                        
                        # Record training loss and average weight/bias for training history
                        self.record_training_history(episode = (episode_no + state_no * num_episodes),
                                                     timestep = total_timesteps,
                                                     loss = current_training_loss)

                        # Record reward
                        self.record_reward(episode = (episode_no + state_no * num_episodes),
                                           timestep = total_timesteps,
                                           reward = reward)
                    
                        # Update Target Network every target_up steps
                        if total_timesteps % target_up == 0:
                            self.training_agent.update_target_model()    
                            
                    # Breaks the episode if ego cars crashes
                    if done:
                        if (self.file_config["save_crash_history_df"] and 
                            counter>=crash_history_size and 
                            (episode_no + state_no * num_episodes) >= crash_history_first_ep):
                            
                            crash_history_df = crash_history_df._append(
                                pd.DataFrame(list(crash_history),
                                             columns=crash_history_df_columns),
                                ignore_index=True)
                            
                            del crash_history
                            with open(self.files["crash_history_df"],'a') as f:                
                                crash_history_df.to_csv(f, index=None, header=f.tell()==0)  
 
                        if ego_state == 1:
                             total_timesteps -= 1
                             counter -= 1
                             print("==============================================")
                             print("================IGNORE CRASH==================")
                             print("==============================================\n")    
                             break
                               
                        print('Episode is forced to stop at step ' + str(counter) + ': '+ \
                              state.translate_car0_state(ego_state))
                        if self.levelk_config["ego"] == 4:
                            print('Dynamic Action Type: '+ dynamic_action_type +'\n' +
                                  'Action Type: '+ action_type +'\n', 
                                  end = '\n')
                        elif 1 <= self.levelk_config["ego"] <= 3:
                            print('Action Type: '+ action_type +'\n',
                                  end = '\n')                      
                        break
                    
                    if ((self.training_config["skip_long_episode"] or 
                         self.training_config["reload_after_long_episode"]) and
                        counter >= max_episode_length):
                        long_episode_flag = True
                        if self.training_config["skip_long_episode"]:
                            print("*********************************************")
                            print("**************SKIP LONG EPISODE**************")
                            print("*********************************************\n")
                        elif self.training_config["reload_after_long_episode"]:
                            print("*****************************************************")
                            print("**************RELOAD AFTER LONG EPISODE**************")
                            print("*****************************************************\n")                            
                        break
                
                if self.file_config["save_ego_df"]:            
                    with open(self.files["ego_df"],'a') as f:                
                        ego_df.to_csv(f, index=None, header=f.tell()==0)
                        del ego_df
                                       
                if not done and ego_reached_end:
                    print("========================================")
                    print("Ego reached end of road at step " + str(counter))
                    if self.levelk_config["ego"] == 4:
                        print('Dynamic Action Type: '+ dynamic_action_type)
                        print('Action Type: '+ action_type)
                    elif 1 <= self.levelk_config["ego"] <= 3:
                        print('Action Type: '+ action_type)
                    print("========================================"+'\n', end = '\n')
                
                # Record collision
                self.record_collision(episode = (episode_no + state_no * num_episodes),
                                      ego_final_state = ego_state)
      
                episode_no += 1
                
                if self.training_config["reload_after_long_episode"] and long_episode_flag:
                    break
            
            if self.training_config["skip_long_episode"] or not long_episode_flag:
                # This updates the Boltzmann temperature of the ego driver
                # Temperature drops down to 1 when state $boltzmann_decay_end_state$
                # is reached
                
                self.training_agent.update_temperature(step = (1.0/boltzmann_decay_end_state))
                # self.training_agent.T = np.maximum(self.training_agent.T * 
                #                                    (1.0/boltzmann_decay_end_state), 
                #                                    1) 
                
                path = self.file_config["path"]
                directory = self.file_config["directory"]
                #Save the weights at the end of each 100 episodes
                fname = './'+path+directory+'/models/model'+str(state_no)
                tfname = './'+path+directory+'/target_weights/target_weight'+str(state_no)+'.h5'
            
                self.training_agent.save(fname,tfname, backup=True)
                self.training_agent.save_memory(path+directory+
                                                "/agent_data/agent_memory"+str(state_no)+".pickle")
                self.training_agent.save_config(total_timesteps, path+directory+
                                                "/agent_data/agent_config"+str(state_no)+".pickle") 
                
                state_no += 1
            else:
                if self.training_config["reload_after_long_episode"]:
                    first_state_reset = state_no
                    self.training_config["retraining_config"] = {"retrain": True,
                                                                 "first_state_reset": first_state_reset} 
                    # Reload last settings for retraining
                    total_timesteps = self._prepare_retraining()
    
def main():
    path = os.path.split(os.getcwd())[0]+"/experiments/some_title/" # Path of the main directory for this experiment
    
    train = True
    simulate = False

    ##########################################
    # PARAMETERS THAT ARE CHANGED FREQUENTLY #
    ##########################################
    # BOTH Training & Simulation
    ego_level = 1 # Level-k Policy for ego (0: Level-0, 1: Level-1, 2: Level-2, 3: Level-3, 4: Dynamic)
    # Level-k Models (for surrounding vehicles)
    # "None" for ego's and unused levels
    models = {1: None, # Level-1 Model
              2: None, # Level-2 Model
              3: None, # Level-3 Model
              4: None} # Dynamic Model
    random_dynamic_strategy = False # Dynamic agent setting: Select strategies randomly 
    dynamic_driving_boltzmann_sampling = False # Dynamic agent setting: Driving actions are sampled via Boltzmann
    ##########################################
    # ONLY Training
    rnd_levk_env_eps = 1 # Dynamic agent setting: Num of episodes in series to create environment with randomly selected level-k strategy
    dynamic_vs = -2 # Dynamic agent setting: -2: Rnd Single, -1: Rnd Mixed
    # Retraining
    retrain = False # Retraining: Restart training from a specified state
    first_state_reset = 95 # Retraining: The model to load is numbered as $first_state_reset - 1$
    ##########################################
    # ONLY Simulation
    sim_save_ego_df = True # Save only ego information into csv file
    sim_ego_models = [95,96,97,98,99] # Ego models to be iterated during the simulation
    sim_vs = [0,1] # Environment type list to be iterated during the simulation
    ##########################################

    ##########################################
    # TRAINING PART
    if train:
        # Level-k Configuration
        distribution_bound = 1.0 # Bound for uniform distribution to select the ratio of different levels in the environment
        boltzmann_sampling = True # If true, agents select action via Boltzmann sampling
        ignore_stopping = True # If True, ignore stopping case
        
        #####################################    
        # TRAINING CONFIGURATION PARAMETERS #
        #####################################
        # Number of states that training will go through
        # At each state, population/Boltzmann Temparature can change and the trained model is saved
        num_state_resets = 100
        num_episodes = 100 # Number of episodes at a state
        target_up = 1000 # Number of steps to be taken to update the target model
        batch_size = 32 # Batch size for experience replay
        replay_start_size = 5000 # Number of frames to be stacked to start experience replay
        stop_sinusoidal_state = 50 # The state to stop sinusoidal population change
        first_phase = 2 # Number of states for the initial warm-up population period
        boltzmann_decay_end_state = 50 # The state to stop Boltzmann temperature decay
        change_car_population_every_ep = 100 # Iterate through car_population list after this many episodes
        random_car_population = False # Randomize population change
        
        # Curriculum method 
        # This phase increases population every $curriculum_skip_every_state$ states
        # from the lowest population to the highest one in $car_population$ list
        # Method is stoped at state $curriculum_end_state$
        enable_curriculum = False
        curriculum_end_state = 48
        curriculum_skip_every_state = 8
    
        car_population = [4,8,12,16,20,24,20,16,12,8] # [16] #  List used for changing population
        ego_lane = -1 # -1: Random / 0: Lane-0 (Ramp) / 1: Lane-1 (Main Road)
        add_car_prob = 0.7 # Add a new car with this probability when enough space is available in the beginning of the road
        bias_lane0vs1 = 0.7 # Add a new car on lane-1 with this probability
        maxcars_onlane0 = 7 # Maximum number of cars on the ramp

        # Long Episode Setting        
        max_episode_length = 300/Params.timestep # After this number of steps, an episode is labeled as long
        reload_after_long_episode = False # If True, reload from the last checkpoint/state when a long episode occurs
        skip_long_episode = True # If true, skip a long episode
    
        # File Configuration Parameters
        save_crash_history_df = True # Save crash history into a csv file
        save_training_df = False # Save training into a csv file
        save_ego_df = True # If check_ego_df_condition returns True, current state is saved
        crash_history_size = 10 # Number of steps to record that occured just before a collision
        crash_history_first_ep = 0  # Start recording crash history at this episode
        
        if ego_level != 4:
            directory = "level"+str(ego_level)+"/training/"
        else:
            directory = "dynamic/training/"
        
        if ego_level <= 3:
            vs = ego_level - 1 # -2: Rnd Single, -1: Rnd Mixed, 0: Level-0, 1: Level-1, 2: Level-2, 3: Level-3, 4: Dynamic
        else:
            vs = dynamic_vs # -2: Rnd Single, -1: Rnd Mixed, 0: Level-0, 1: Level-1, 2: Level-2, 3: Level-3, 4: Dynamic
            
        train_models = {1: None, 2:None, 3:None, 4:None}
        if 1 <= vs <= 3:
            train_models[vs] = models[vs]
        elif -2 <= vs <= -1:
            train_models[1] = models[1]
            train_models[2] = models[2]
            train_models[3] = models[3]
        
        # Level-k Configuration Dictionary for Training
        levelk_config = {"ego": ego_level,
                          "vs": vs,
                          "models":{1: train_models[1],
                                    2: train_models[2],
                                    3: train_models[3],
                                    4: train_models[4]},
                          "boltzmann_sampling": boltzmann_sampling,
                          "dynamic_driving_boltzmann_sampling": dynamic_driving_boltzmann_sampling,
                          "random_dynamic_strategy": random_dynamic_strategy,
                          "distribution_bound": distribution_bound}
    
        # Curriculum Configuration Dictionary for Training
        curriculum_config = {"enable_curriculum": enable_curriculum,
                              "curriculum_end_state": curriculum_end_state,
                              "curriculum_skip_every_state": curriculum_skip_every_state}
        
        # Retraining Configuration Dictionary
        retraining_config = {"retrain": retrain,
                              "first_state_reset": first_state_reset} 
        
        # Training Configuration Dictionary for Training
        training_config = {"num_episodes": num_episodes,
                           "rnd_levk_env_eps": rnd_levk_env_eps,
                            "random_car_population": random_car_population,
                            "num_state_resets": num_state_resets,
                            "target_up": target_up,
                            "batch_size": batch_size,
                            "replay_start_size": replay_start_size,
                            "stop_sinusoidal_state": stop_sinusoidal_state,
                            "first_phase": first_phase,
                            "change_car_population_every_ep": change_car_population_every_ep,
                            "boltzmann_decay_end_state": boltzmann_decay_end_state,
                            "skip_long_episode": skip_long_episode,
                      "car_population": car_population,
                            "ego_lane": ego_lane,
                            "add_car_prob": add_car_prob,
                            "bias_lane0vs1": bias_lane0vs1,
                            "maxcars_onlane0": maxcars_onlane0,
                            "max_episode_length": max_episode_length,
                                "reload_after_long_episode": reload_after_long_episode,
                            "retraining_config": retraining_config,
                            "curriculum_config": curriculum_config,
                            "ignore_stopping":ignore_stopping}
        
        # File Configuration Dictionary for Training
        file_config = {"path": path,
                        "directory": directory,
                        "save_crash_history_df": save_crash_history_df,
                        "save_training_df": save_training_df,
                        "save_ego_df": save_ego_df,
                        "crash_history_size": crash_history_size,
                        "crash_history_first_ep": crash_history_first_ep}
        
        training = Training(levelk_config = levelk_config,
                          training_config = training_config,
                          file_config = file_config)
        
        training.run()

    ##########################################
    # SIMULATION PART
    if simulate: 
        if random_dynamic_strategy:
            sim_ego_models = [None]
            
        for model_ego in sim_ego_models:
            for vs_temp in sim_vs: # -2: Rnd Single, -1: Rnd Mixed, 0: Level-0, 1: Level-1, 2: Level-2, 3: Level-3, 4: Dynamic

                # Level-k Configuration Parameters
                distribution_bound = 1.0 # Bound for uniform distribution to select the ratio of different levels
                boltzmann_sampling = True # If true, agents select action via Boltzmann sampling

                # Simulation Configuration Parameters           
                num_episodes = 500 # Number of episodes for each population group
                car_population = [12,16,20,24] # List used for changing population
                ego_lane = -1 # -1: Random / 0: Lane-0 (Ramp) / 1: Lane-1 (Main Road)
                add_car_prob = 0.7 # Add a new car with this probability when enough space is available in the beginning of the road
                bias_lane0vs1 = 0.7 # Add a new car on lane-1 with this probability
                maxcars_onlane0 = 7 # Maximum number of cars on the ramp
                max_episode_length = 300/Params.timestep # Max number of steps to stop an episode
                
                save_sim_df = False # Save simulation into csv file


                sim_models = {1: None, 2:None, 3:None, 4:None}
                if 1 <= vs_temp <= 4:
                    sim_models[vs_temp] = models[vs_temp]
                elif -2 <= vs_temp <= -1:
                    sim_models[1] = models[1]
                    sim_models[2] = models[2]
                    sim_models[3] = models[3]
                    
                sim_models[ego_level] = model_ego
                # File Configuration Parameters
                if ego_level != 4:
                    directory = "level"+str(ego_level)+"/simulation/"
                else:
                    directory = "dynamic/simulation/"
                
                # Level-k Configuration Dictionary for Simulation
                levelk_config = {"ego": ego_level,
                                  "vs": vs_temp,
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
                                "save_ego_df": sim_save_ego_df}
                
                sim = Simulation(levelk_config = levelk_config,
                                  simulation_config = simulation_config,
                                  file_config = file_config)
                
                sim.run()
                
                analyzer = SimulationAnalyzer(analyze_config = sim.get_analyze_config()) 
                analyzer.analyze()        
            
if __name__ == '__main__':
    main()
