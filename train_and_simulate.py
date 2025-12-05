"""
This file is used to train and simulate the agents in the highway merging environment.

To start off the training or simulation process, set the 'train' or 'simulate' variables to True respectively.
To train level 1 agent, set 'ego_level' to 1. Similarly, set it to 2, 3, or 4 for level 2, level 3, or dynamic agents.
For the surrounding vehicles, specify the models for level-k agents in the 'models' dictionary.
After training, make sure the training went well by running the merge_and_plot.py file to visualize the results.
You may directly take the 99th model of each level as the trained model for that level,
or simulate the last 5-10 models with the lower level models to see which model performs better.
To do simulation, set the 'sim_vs' variable to the desired level-k environment type, and 'sim_ego_models' to the desired model of the ego agent.
To do retraining, set 'retrain' to True and fill in the 'first_state_reset' variable. This is useful when you want to continue training from a previously trained model.

Example configurations:
1. To train a level-1 agent against level-0 agents:
   - Set 'train' to True
   - Set 'ego_level' to 1
   - Set 'models' to {1: None, 2: None, 3: None, 4: None}
2. To simulate a level-2 agent using model 95 against level-1 agents using model 99:
   - Set 'simulate' to True
   - Set 'ego_level' to 2
   - Set 'models' to {1: 99, 2: None, 3: None, 4: None}
   - Set 'sim_vs' to [1]
   - Set 'sim_ego_models' to [95]
3. To retrain a dynamic agent from model 10:
   - Set 'train' to True
   - Set 'ego_level' to 4
   - Set 'models' to {1: 99, 2: 99, 3: 99, 4: None}
   - Set 'retrain' to True
   - Set 'first_state_reset' to 11

"""

#def train_and_simulate_time():
from src.Training import Training
from src.Simulation import Simulation
from Params import Params
from src.SimulationAnalyzer import SimulationAnalyzer
import warnings
import os
warnings.simplefilter(action='ignore', category=FutureWarning) #not to see futurewarning warning
cwd = os.getcwd()

path = "ATR_Original\\experiments\\some_title\\" # Path of the main directory for this experiment no need to change for me 

train = False
simulate = True

##########################################
# PARAMETERS THAT ARE CHANGED FREQUENTLY #
##########################################
# BOTH Training & Simulation
ego_level = 1 # Level-k Policy for ego (0: Level-0, 1: Level-1, 2: Level-2, 3: Level-3, 4: Dynamic)
# Level-k Models (for surrounding vehicles)
# "None" for ego's and unused levels
models = {1: 98, # Level-1 Model
        2: None, # Level-2 Model
        3: None, # Level-3 Modelwe   
        4: None} # Dynamic Model
random_dynamic_strategy = False # Dynamic agent setting: Select strategies randomly 
dynamic_driving_boltzmann_sampling = True # Dynamic agent setting: Driving actions are sampled via Boltzmann
##########################################
# ONLY Training
rnd_levk_env_eps = 1 # Dynamic agent setting: Num of episodes in series to create environment with randomly selected level-k strategy
dynamic_vs = -1 # Dynamic agent setting: -2: Rnd Single, -1: Rnd Mixed
# Retraining
retrain = True # Retraining: Restart training from a specified state
first_state_reset = 85 # Retraining: The model to load is numbered as $first_state_reset - 1$
                      # for instance, if you want to retrain from model 10, put 11 here
##################  ########################
# ONLY Simulation
sim_save_ego_df = True # Save only ego information into csv file
sim_ego_models = [97] # Ego models to be iterated during the simulation / model no 99 etc.
sim_vs = [0] # Environment type list to be iterated during the simulation with 1,2,3 etc.
##########################################

###################################                #######
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
    # This phase increases population every $curriculum_ski p_every_state$ states
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
        train_models[vs] = models[vs]   #here, vs is ego-1. models[vs] is ego-1 level model tobe used. Inıtıalized None trainmodels [ego-1] is model of level k-1
    elif -2 <= vs <= -1:
        train_models[1] = models[1]
        train_models[2] = models[2]
        train_models[3] = models[3]     #if dynamic, load all models for all levels 

    
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
                        "car_population": car_population,
                        "ego_lane": ego_lane,
                        "add_car_prob": add_car_prob,
                        "bias_lane0vs1": bias_lane0vs1,
                        "maxcars_onlane0": maxcars_onlane0,
                        "max_episode_length": max_episode_length,
                        "num_state_resets": num_state_resets,
                        "target_up": target_up,
                        "batch_size": batch_size,
                        "replay_start_size": replay_start_size,
                        "stop_sinusoidal_state": stop_sinusoidal_state,
                        "first_phase": first_phase,
                        "change_car_population_every_ep": change_car_population_every_ep,
                        "boltzmann_decay_end_state": boltzmann_decay_end_state,
                        "skip_long_episode": skip_long_episode,
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
    warnings.filterwarnings('ignore', category=FutureWarning, message='.*DataFrame concatenation with empty or all-NA entries.*')

    #there expected a bottleneck + file is not overrided for retraining purposes
    training.run()

##########################################
# SIMULATION PART
if simulate: 
    if random_dynamic_strategy:
        sim_ego_models = [None]     #in which model of the selected level you are simulating
        
    for model_ego in sim_ego_models:
        for vs_temp in sim_vs: # -2: Rnd Single, -1: Rnd Mixed, 0: Level-0, 1: Level-1, 2: Level-2, 3: Level-3, 4: Dynamic
            #sim_vs was the environment level
            # Level-k Configuration Parameters
            distribution_bound = 1.0 # Bound for uniform distribution to select the ratio of different levels
            boltzmann_sampling = True # If true, agents select action via Boltzmann sampling

            # Simulation Configuration Parameters           
            num_episodes = 250 # Number of episodes for each population group
            car_population = [ 12, 16, 20, 24 ] # List used for changing population
            ego_lane = -1 # -1: Random / 0: Lane-0 (Ramp) / 1: Lane-1 (Main Road)
            add_car_prob = 0.7 # Add a new car with this probability when enough space is available in the beginning of the road
            bias_lane0vs1 = 0.7 # Add a new car on lane-1 with this probability
            maxcars_onlane0 = 7 # Maximum number of cars on the ramp
            max_episode_length = 300/Params.timestep # Max number of steps to stop an episode
            
            save_sim_df = True # Save simulation into csv file


            sim_models = {1: None, 2:None, 3:None, 4:None}
            # if 1 <= vs_temp <= 4:
            #     sim_models[vs_temp] = models[vs_temp]
            # elif -2 <= vs_temp <= -1:
            #     sim_models[1] = models[1]
            #     sim_models[2] = models[2]
            #     sim_models[3] = models[3]

            sim_models = {
                1: models[1],
                2: models[2],
                3: models[3],
                4: models[4]
            }
                
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
            
            warnings.filterwarnings(action='ignore', category=FutureWarning, module='pandas.core.frame')
            sim.run()
            
            analyzer = SimulationAnalyzer(analyze_config = sim.get_analyze_config()) 
            analyzer.analyze()   
"""                       
def main():
import cProfile
import pstats
with cProfile.Profile() as pr:
    train_and_simulate_time()
stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.print_stats
"""