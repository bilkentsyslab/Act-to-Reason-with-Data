#This file implements the simulation analyzer
# © 2020 Cevahir Köprülü All Rights Reserved

import numpy as np
import pandas as pd
import pickle
import Params
import os

class SimulationAnalyzer():
    
    def __init__(self,analyze_config):
        self.analyze_config = analyze_config
    
    def analyze_simdf(self, sim_df_file, analysis_df_file, 
                      num_episodes, crash_eps_pickle, succ_eps_pickle, 
                      long_eps_pickle, car_population):
        """
        Simulation dataframe is analyzed here.

        Parameters
        ----------
        sim_df_file : string
            Pathway of a simulation dataframe.
        analysis_df_file : string
            Pathway of a simulation dataframe.
        num_episodes : int
            Number of episodes for each population group.
        crash_eps_pickle : string
            Pathway for a pickle file to record episodes that ends with ego related collisions.
        succ_eps_pickle : string
            Pathway for a pickle file to record successful episodes.
        long_eps_pickle : string
            Pathway for a pickle file to record long episodes.
        car_population : list
            List of population groups.

        Returns
        -------
        None.

        """
        sim_df = pd.read_csv(sim_df_file)
        
        ego = sim_df.loc[sim_df['Car_ID']==0]
        ego_last_step = ego[(ego.duplicated(subset='Episode',keep='last')==False)];
        ego_long_eps = ego_last_step.loc[ego_last_step['State']==-1].Episode
        ego_succ_eps = ego_last_step.loc[ego_last_step['State']==4].Episode;
        ego_ignore_eps = ego_last_step.loc[ego_last_step['State']==1].Episode;
        
        long_eps = set(ego_long_eps.tolist())
        succ_eps = set(ego_succ_eps.tolist())
        ignore_eps = set(ego_ignore_eps.tolist())
        eps = set(range(0,7*num_episodes))
        crash_eps = eps - (long_eps.union(succ_eps)).union(ignore_eps)
        
        if len(crash_eps) != 0:
            with open(crash_eps_pickle, "wb") as output_file:
                pickle.dump(crash_eps, output_file)

        if len(succ_eps) != 0:
            with open(succ_eps_pickle, "wb") as output_file:
                pickle.dump(succ_eps, output_file)
             
        if len(long_eps) != 0:
            with open(long_eps_pickle, "wb") as output_file:
                pickle.dump(long_eps, output_file)

        states = Params.Params.car0_states
        crash_freq = np.zeros((len(car_population),4,2),dtype=int)
        
        for i,crash in ego_last_step.iterrows():
            if crash["State"] != -1 and crash["State"] != 4:
                crash_type = int(crash["State"])
                episode = crash["Episode"]
                j = int(episode/num_episodes)
                crash_freq[j,crash_type,int(crash["Lane_ID"])] += 1
        
        analysis = []
        for i in range(crash_freq.shape[0]):              
            numcars = car_population[i]
            print("*********************************************")
            print("Number of cars: " + str(numcars) + "\n")
            for j in range(2):
                analysis.append([numcars,crash_freq[i,0,j],crash_freq[i,1,j],
                                 crash_freq[i,2,j],crash_freq[i,3,j],j,
                                 np.sum(crash_freq[i,:,j])])
                
                print("Lane_ID: " + str(j))
                print(states[0]+": " + str(crash_freq[i,0,j]))
                print(states[1]+": " + str(crash_freq[i,1,j]))
                print(states[2]+": " + str(crash_freq[i,2,j]))
                print(states[3]+": " + str(crash_freq[i,3,j]) + "\n")
        
        analysis.append(["Sum on Lane 0",np.sum(crash_freq[:,0,0]),
                         np.sum(crash_freq[:,1,0]),np.sum(crash_freq[:,2,0]),
                         np.sum(crash_freq[:,3,0]),0,np.sum(crash_freq[:,:,0])])
        analysis.append(["Sum on Lane 1",np.sum(crash_freq[:,0,1]),
                         np.sum(crash_freq[:,1,1]),np.sum(crash_freq[:,2,1]),
                         np.sum(crash_freq[:,3,1]),1,np.sum(crash_freq[:,:,1])])
        analysis.append(["Sum",np.sum(crash_freq[:,0,:]),np.sum(crash_freq[:,1,:]),
                         np.sum(crash_freq[:,2,:]),np.sum(crash_freq[:,3,:]),
                         "Both",np.sum(crash_freq[:,:,:])])
        
        analysis_df = pd.DataFrame(analysis, columns = ['Numcars', str(states[0]),
                                                        str(states[1]),str(states[2]),
                                                        str(states[3]),'Lane','Sum'])
        with open(analysis_df_file,'a') as f:                
            analysis_df.to_csv(f, index=None, header=f.tell()==0)        
    
    def analyze_coldf(self, col_df_file, analysis_df_file, car_population, num_episodes):
        """
        Collision dataframe is analyzed here.

        Parameters
        ----------
        col_df_file : string
            Pathway of a collision dataframe.
        analysis_df_file : string
            Pathway to save the analysis results of the collision dataframe.
        car_population : list
            List of population groups.
        num_episodes : int
            Number of episodes for each population group.

        Returns
        -------
        None.

        """
        col_df = pd.read_csv(col_df_file)
              
        states = Params.Params.car0_states
        crash_freq = np.zeros((len(car_population),4,2),dtype=int)
        for i,crash in col_df.iterrows():
            crash_type = int(crash["State"])
            episode = crash["Episode"]
            j = int(episode/num_episodes)
            crash_freq[j,crash_type,int(crash["lane"])] += 1
        
        analysis = []
        for i in range(crash_freq.shape[0]):
            
            numcars = car_population[i]
            print("*********************************************")
            print("Number of cars: " + str(numcars) + "\n")
            for j in range(2):
                analysis.append([numcars,crash_freq[i,0,j],crash_freq[i,1,j],
                                 crash_freq[i,2,j],crash_freq[i,3,j],j,
                                 np.sum(crash_freq[i,:,j])])
                print("Lane_ID: " + str(j))
                print(states[0]+": " + str(crash_freq[i,0,j]))
                print(states[1]+": " + str(crash_freq[i,1,j]))
                print(states[2]+": " + str(crash_freq[i,2,j]))
                print(states[3]+": " + str(crash_freq[i,3,j]) + "\n")
        
        analysis.append(["Sum on Lane 0",np.sum(crash_freq[:,0,0]),
                         np.sum(crash_freq[:,1,0]),np.sum(crash_freq[:,2,0]),
                         np.sum(crash_freq[:,3,0]),0,np.sum(crash_freq[:,:,0])])
        analysis.append(["Sum on Lane 1",np.sum(crash_freq[:,0,1]),
                         np.sum(crash_freq[:,1,1]),np.sum(crash_freq[:,2,1]),
                         np.sum(crash_freq[:,3,1]),1,np.sum(crash_freq[:,:,1])])
        analysis.append(["Sum",np.sum(crash_freq[:,0,:]),np.sum(crash_freq[:,1,:]),
                         np.sum(crash_freq[:,2,:]),np.sum(crash_freq[:,3,:]),
                         "Both",np.sum(crash_freq[:,:,:])])
        
        analysis_df = pd.DataFrame(analysis, columns = ['Numcars', str(states[0]),
                                                        str(states[1]),str(states[2]),
                                                        str(states[3]),'Lane','Sum'])
        with open(analysis_df_file,'a') as f:                
            analysis_df.to_csv(f, index=None, header=f.tell()==0)   
                
    def analyze_egodf(self, ego_df_file, ego_df_analysis_file, ego_level, 
                      car_population, num_episodes):
        """
        Ego dataframe is analyzed here.

        Parameters
        ----------
        ego_df_file : string
            Pathway of the ego dataframe to be analyzed.
        ego_df_analysis_file : string
            Pathway of the analysis dataframe to be saved.
        ego_level : int
            Level-k strategy of the ego.
        car_population : list
            List of the population groups.
        num_episodes : int
            Number of episodes for each population group.

        Returns
        -------
        None.

        """
        if os.path.exists(ego_df_analysis_file):
            os.remove(ego_df_analysis_file) 
        
        ego_df = pd.read_csv(ego_df_file)
        
        if ego_level == 4:
            ego_freq = np.zeros((len(car_population),3,2),dtype=int)
            for i,ego_row in ego_df.iterrows():
                episode = ego_row["Episode"]
                j = int(episode/num_episodes)
                ego_freq[j,int(ego_row["Dynamic_Action"]),int(ego_row["lane"])] += 1
            
            analysis_ego = []
            for i in range(ego_freq.shape[0]):
                
                numcars = car_population[i]
                for j in range(2):
                    analysis_ego.append([numcars,
                                         ego_freq[i,0,j],
                                         ego_freq[i,1,j],
                                         ego_freq[i,2,j],
                                         j,
                                         np.sum(ego_freq[i,:,j])])
            
            analysis_ego.append(["Sum on Lane 0",
                                 np.sum(ego_freq[:,0,0]),
                                 np.sum(ego_freq[:,1,0]),
                                 np.sum(ego_freq[:,2,0]),
                                 0,
                                 np.sum(ego_freq[:,:,0])])
            
            analysis_ego.append(["Sum on Lane 1",
                                 np.sum(ego_freq[:,0,1]),
                                 np.sum(ego_freq[:,1,1]),
                                 np.sum(ego_freq[:,2,1]),
                                 1,
                                 np.sum(ego_freq[:,:,1])])
            
            analysis_ego.append(["Sum",
                                 np.sum(ego_freq[:,0,:]),
                                 np.sum(ego_freq[:,1,:]),
                                 np.sum(ego_freq[:,2,:]),
                                 "Both",
                                 np.sum(ego_freq[:,:,:])])
            
            ego_df_analysis = pd.DataFrame(analysis_ego, columns = ['Numcars',
                                                                    'Level-1',
                                                                    'Level-2',
                                                                    'Level-3',
                                                                    'Lane',
                                                                    'Sum'])
            with open(ego_df_analysis_file,'a') as f:                
                ego_df_analysis.to_csv(f, index=None, header=f.tell()==0)
        else:
            ego_freq = np.zeros((len(car_population),6,2),dtype=int)
            for i,ego_row in ego_df.iterrows():
                episode = ego_row["Episode"]
                j = int(episode/num_episodes)
                ego_freq[j,int(ego_row["Action"]),int(ego_row["lane"])] += 1
            
            analysis_ego = []
            for i in range(ego_freq.shape[0]):
                
                numcars = car_population[i]
                for j in range(2):
                    analysis_ego.append([numcars,
                                         ego_freq[i,0,j],
                                         ego_freq[i,1,j],
                                         ego_freq[i,2,j],
                                         ego_freq[i,3,j],
                                         ego_freq[i,4,j],
                                         ego_freq[i,5,j],
                                         j,
                                         np.sum(ego_freq[i,:,j])])
            
            analysis_ego.append(["Sum on Lane 0",
                                 np.sum(ego_freq[:,0,0]),
                                 np.sum(ego_freq[:,1,0]),
                                 np.sum(ego_freq[:,2,0]),
                                 np.sum(ego_freq[:,3,0]),
                                 np.sum(ego_freq[:,4,0]),
                                 np.sum(ego_freq[:,5,0]),
                                 0,
                                 np.sum(ego_freq[:,:,0])])
            
            analysis_ego.append(["Sum on Lane 1",
                                 np.sum(ego_freq[:,0,1]),
                                 np.sum(ego_freq[:,1,1]),
                                 np.sum(ego_freq[:,2,1]),
                                 np.sum(ego_freq[:,3,1]),
                                 np.sum(ego_freq[:,4,1]),
                                 np.sum(ego_freq[:,5,1]),
                                 1,
                                 np.sum(ego_freq[:,:,1])])
            
            analysis_ego.append(["Sum",
                                 np.sum(ego_freq[:,0,:]),
                                 np.sum(ego_freq[:,1,:]),
                                 np.sum(ego_freq[:,2,:]),
                                 np.sum(ego_freq[:,3,:]),
                                 np.sum(ego_freq[:,4,:]),
                                 np.sum(ego_freq[:,5,:]),
                                 "Both",
                                 np.sum(ego_freq[:,:,:])])
            
            ego_df_analysis = pd.DataFrame(analysis_ego, columns = ['Numcars',
                                                                    'Maintain',
                                                                    'Accel',
                                                                    'Decel',
                                                                    'Hard-Accel',
                                                                    'Hard-Decel',
                                                                    'Merge',
                                                                    'Lane',
                                                                    'Sum'])
            with open(ego_df_analysis_file,'a') as f:                
                ego_df_analysis.to_csv(f, index=None, header=f.tell()==0)           
    
    def analyze(self):
        """
        This function enables the analysis of simulation/ego/collision dataframes

        Returns
        -------
        None.

        """        
        num_episodes = self.analyze_config["num_episodes"]        
        car_population = self.analyze_config["car_population"]
        ego_level = self.analyze_config["ego"]
        sim_df_file = self.analyze_config["files"]["sim_df_file"]
        ego_df_file = self.analyze_config["files"]["ego_df_file"]
        col_df_file = self.analyze_config["files"]["col_df_file"]
        analysis_df_file = self.analyze_config["files"]["analysis_df_file"]
        ego_df_analysis_file = self.analyze_config["files"]["ego_df_analysis_file"] 
        long_eps_pickle = self.analyze_config["pickle"]["long_eps_pickle"]
        succ_eps_pickle = self.analyze_config["pickle"]["succ_eps_pickle"]
        crash_eps_pickle = self.analyze_config["pickle"]["crash_eps_pickle"]
        
        analyze_egodf = ego_df_file != None
        analyze_simdf = sim_df_file != None
        analyze_coldf = (sim_df_file == None) and os.path.exists(col_df_file)

        
        if os.path.exists(analysis_df_file):
            os.remove(analysis_df_file)
        
        if analyze_simdf:
            self.analyze_simdf(sim_df_file = sim_df_file,
                               analysis_df_file = analysis_df_file,
                               num_episodes = num_episodes, 
                               crash_eps_pickle = crash_eps_pickle, 
                               succ_eps_pickle = succ_eps_pickle, 
                               long_eps_pickle = long_eps_pickle, 
                               car_population = car_population)
                
        if analyze_coldf:  
            self.analyze_coldf(col_df_file = col_df_file, 
                               analysis_df_file = analysis_df_file, 
                               car_population = car_population, 
                               num_episodes = num_episodes)
        
        if analyze_egodf:
            self.analyze_egodf(ego_df_file = ego_df_file,
                               ego_df_analysis_file = ego_df_analysis_file,
                               ego_level = ego_level,
                               car_population = car_population,
                               num_episodes = num_episodes)
            
def main():
    path = "data_2020_10_01_2/"
    directory = "level1/simulation/"
    ego_level = 1
    analyze_config = {"files":{"sim_df_file": None,
                                "ego_df_file": path+directory+\
                                    "simulation_level_1_m59_vs_L0_7x500eps_ego.csv",
                                "col_df_file":path+directory+\
                                    "simulation_level_1_m59_vs_L0_7x500eps_col.csv",
                                "analysis_df_file": path+directory+\
                                    "simulation_level_1_m59_vs_L0_7x500eps_analysis.csv",
                                "ego_df_analysis_file": path+directory+\
                                    "simulation_level_1_m59_vs_L0_7x500eps_ego_analysis.csv"},
                       "pickle":{"long_eps_pickle": path+directory+\
                                 "simulation_level_1_m54_vs_L0_and_L1_m54_and_L2_m58_and_L3_m58_7x20eps_long_eps.pickle",
                                 "succ_eps_pickle": path+directory+\
                                     "simulation_level_1_m54_vs_L0_and_L1_m54_and_L2_m58_and_L3_m58_7x20eps_successful_eps.pickle",
                                 "crash_eps_pickle": path+directory+"\
                                     simulation_level_1_m54_vs_L0_and_L1_m54_and_L2_m58_and_L3_m58_7x20eps_crash_eps.pickle"},
                       "num_episodes": 500,
                       "car_population": [4,8,12,16,20,24,28],
                       "ego":ego_level}
    
    analyzer = SimulationAnalyzer(analyze_config = analyze_config) 
    analyzer.analyze()

if __name__ == '__main__':
    main()