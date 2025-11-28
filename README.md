````markdown
# I-80 Trajectory Simulation and Model Calibration Project 🚗💨

This project involves the development of **Level-K** and **Dynamic Agent** models for vehicle merging behavior, followed by a data calibration and verification process using I-80 trajectory data.

---

## Setup and Environment

Before running any scripts, you must set up the required Python environment using `conda`.

1.  **Create the Environment:**
    ```bash
    conda env create -f environment.yml
    ```
2.  **Activate the Environment:**
    ```bash
    conda activate atr
    ```

---

## 🛠️ Part 1: Model Building and Basic Simulation (Level-K & Dynamic)

This part focuses on building and simulating the Level-K (K=1, 2, 3) and uncalibrated Dynamic Agent models.

### 1.1 Run Model Training and Simulation

Execute the main script to train and simulate the core models.

* Run the script:
    ```bash
    python train_and_simulate.py
    ```
    This script handles the building and initial simulation runs for the **Level-1, Level-2, Level-3, and baseline Dynamic Agents**.

### 1.2 Analyze Training Results

After the simulation, run the following script to merge and plot the graphs related to the training process (e.g., convergence, loss).

* Run the script:
    ```bash
    python merge_and_plot.py
    ```

### 1.3 Visual Verification (Optional)

If you wish to visualize the training and simulation processes, run the Graphical User Interface (GUI).

* Run the GUI:
    ```bash
    python HighwayMergingGUI.py
    ```

---

## Part 2: Data Calibration and Verification

This section outlines the sequential steps required to process the raw I-80 trajectory data, generate simulation scenarios, **calibrate the dynamic agent**, and run various simulation models against the ground truth for evaluation.

### 2.1 Data Pre-processing: Scenario Generation

This initial stage filters the raw dataset and converts full vehicle trajectories into structured, fixed-duration scenarios and historical data suitable for agent training/calibration. 

| Step | Script | Description | Output Files & Location |
| :--- | :--- | :--- | :--- |
| **2.1.1** | `create_ego_vehicles.py` | Filters raw data to select non-truck vehicles that perform a clean merge from **Lane 7 to Lane 6**. Also splits ego IDs into train/test sets. | `data/ego_ids/*.csv`, `data/vehicles/ego_vehicle_[ID]_with_env.csv` |
| **2.1.2** | `create_timed_episodes.py` | Slices the full episode files into standardized **5-second scenarios** with a preceding **1-second history**, filtering for data quality. | `data/5sec_scenarios/scenarios/*.csv`, `data/5sec_scenarios/histories/*.csv` |
| **2.1.3** | `create_dynamic_histories.py` | Transforms the 1-second history files into a normalized **State-Action** vector format required by the dynamic agent. | `data/5sec_scenarios/dynamic_histories.pickle` |

---

### 2.2 Agent Training and Evaluation

This stage runs the simulations for various agent models, using the generated scenarios and histories, and evaluates their performance metrics.

#### 2.2.1 Calibrated Dynamic Agent

1.  **Training (Calibration):** In `data_train_and_simulate.py`, set **`Train = True`** and `Simulate = False`. Run the script to **calibrate** the dynamic agent with the extracted historical data.
2.  **Simulation:** In `data_train_and_simulate.py`, set **`Train = False`** and `Simulate = True`. Run the script to simulate the **calibrated** dynamic agent.
3.  **Analysis:** Run `frechet_analyzer.py` to record performance metrics (specifically, Frechet distances).

---

#### 2.2.2 Baseline Agent Simulations

| Script | Description | Setup | Analysis |
| :--- | :--- | :--- | :--- |
| `data_simulate_original_dynamic.py` | Simulates the **non-calibrated** dynamic agent (baseline behavior). | Run script. | Run `frechet_analyzer.py`. |
| `data_simulate_levelk.py` | Simulates the **Level-K** model (e.g., K=1, 2, or 3). | Set the desired value for **`k`** (e.g., `1`, `2`, or `3`) directly in the script. | Run `frechet_analyzer.py`. |
| `data_simulate_IDM.py` | Simulates the classical **IDM + MOBIL** traffic model. | Run script. | Run `frechet_analyzer.py`. |
````

Would you like me to generate a brief summary for the top of the README?