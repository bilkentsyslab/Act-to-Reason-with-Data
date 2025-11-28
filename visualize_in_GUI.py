from SimulationAnalyzer import SimulationAnalyzer

path = "ATR_Original/experiments/some_title/level2/simulation/"
ego_level = 2
analyze_config = {"files":{"sim_df_file": None,
                            "ego_df_file": path+\
                                "simulation_level_2_m80_vs_L1_m94_4x500eps_ego.csv",
                            "col_df_file":path+\
                                "simulation_level_2_m80_vs_L1_m94_4x500eps_col.csv",
                            "analysis_df_file": path+\
                                "simulation_level_2_m80_vs_L1_m94_4x500eps_analysis.csv",
                            "ego_df_analysis_file": path+\
                                "simulation_level_2_m80_vs_L1_m94_4x500eps_ego_analysis.csv"},
                   "pickle":{"long_eps_pickle": None,
                             "succ_eps_pickle": None,
                             "crash_eps_pickle": None},
                   "num_episodes": 500,
                   "car_population": [4,8,12,16,20,24,28],
                   "ego":ego_level}

analyzer = SimulationAnalyzer(analyze_config = analyze_config) 
analyzer.analyze()
