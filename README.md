This document outlines the sequential steps required to process the raw I-80 trajectory data, generate simulation scenarios, calibrate the dynamic agent, and run various simulation models for evaluation.

### 1. Data Pre-processing: Scenario Generation

This initial stage filters the raw dataset and converts full vehicle trajectories into structured, fixed-duration scenarios and historical data suitable for agent training.

| Step | Script | Description | Output Files & Location |
| :--- | :--- | :--- | :--- |
| **1.1** | `create_ego_vehicles.py` | Filters raw data to select non-truck vehicles that perform a clean merge from **Lane 7 to Lane 6**. Also splits ego IDs into train/test sets. | `data/ego_ids/*.csv`, `data/vehicles/ego_vehicle_[ID]_with_env.csv` |
| **1.2** | `create_timed_episodes.py` | Slices the full episode files into standardized **5-second scenarios** with a preceding **1-second history**, filtering for data quality. | `data/5sec_scenarios/scenarios/*.csv`, `data/5sec_scenarios/histories/*.csv` |
| **1.3** | `create_dynamic_histories.py` | Transforms the 1-second history files into a normalized **State-Action** vector format required by the dynamic agent. | `data/5sec_scenarios/dynamic_histories.pickle` |

---

### 2. Agent Training and Evaluation

This stage runs the simulations for various agent models, using the generated scenarios and histories.

#### 2.1 Calibrated Dynamic Agent

1.  **Training:** In `data_train_and_simulate.py`, set **`Train = True`** and `Simulate = False`. Run the script to **calibrate** the dynamic agent with the extracted historical data.
2.  **Simulation:** In `data_train_and_simulate.py`, set **`Train = False`** and `Simulate = True`. Run the script to simulate the **calibrated** dynamic agent.
3.  **Analysis:** Run `frechet_analyzer.py` to record performance metrics (Frechet distances).

---

#### 2.2 Baseline Agent Simulations

| Script | Description | Setup | Analysis |
| :--- | :--- | :--- | :--- |
| `data_simulate_original_dynamic.py` | Simulates the **non-calibrated** dynamic agent (baseline behavior). | Run script. | Run `frechet_analyzer.py`. |
| `data_simulate_levelk.py` | Simulates the **Level-K** model (e.g., K=1, 2, or 3). | Set **`k`** to the desired value (e.g., `1`, `2`, or `3`) in the script. | Run `frechet_analyzer.py`. |
| `data_simulate_IDM.py` | Simulates the classical **IDM + MOBIL** traffic model. | Run script. | Run `frechet_analyzer.py`. |
