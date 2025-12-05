#This implemets a class for the state of the highway merging environment
# © 2020 Cevahir Köprülü All Rights Reserved

from Params import Params
import random
import numpy as np 

class State:
    def __init__(self, *args):
        if isinstance(args[0], int):
            self.init(args[0],args[1],args[2],args[3], args[4])
        else :
            self.init1(args[0])
    #Initializes the environment by adding cars to the road and selecting positions
    def init(self, numcars, ego_lane, bias_lane0vs1, maxcars_onlane0, levk_distribution):
        
        # cars_info: columns are car instances 
        # which include the following properties:
        # 0: position_x: float
        # 1: prev_position_x: float
        # 2: velocity_x: float
        # 3: acceleration_x: float
        self.cars_info = np.zeros((4,numcars))
        
        # cars_info2: columns are car instances which include the following properties:
        # 0: lane: int       
        # 1: current_action: can be int, but Action object
        # 2: prev_action: can be int, but Action object
        # 3: effort: int
        # 4: lane_changed: bool, can be int
        # 5: level_k: int => k=0,1,2,3,4 / 4: Dynamic
        self.cars_info2 = np.zeros((7,numcars),dtype=int)
        
        self.numcars = [0,0]
        self.maxcars_onlane0 = maxcars_onlane0
        self.bias_lane0vs1 = bias_lane0vs1
        self.isHeavy = False
        self.levk_distribution = levk_distribution
        self.numcars_levk = [0,0,0,0,0]
        self.initialize_environment(ego_lane)

    # Initializes the environment by copying given state
    def init1(self, copy):
        self.cars_info = copy.cars_info
        self.cars_info2 = copy.cars_info2
        self.numcars =  copy.numcars
        self.maxcars_onlane0 = copy.maxcars_onlane0
        self.bias_lane0vs1 = copy.bias_lane0vs1
        self.isHeavy = copy.isHeavy
        self.levk_distribution = copy.levk_distribution
        self.numcars_levk = copy.numcars_levk
        
    #Selects appropriote positions for each car
    def initialize_environment(self,ego_lane):
        # Assign ego lane
        if ego_lane == -1:
            self.cars_info2[0,0] = int(random.uniform(0.0, 1.0) < 0.5)
        else:
            self.cars_info2[0,0] = ego_lane
        
        # Assign ego position with respect to its lane
        if self.cars_info2[0,0] == 0:
            self.numcars[0] += 1
            self.cars_info[0,0] = Params.start_onramp
            self.cars_info[1,0] = Params.start_onramp
            self.cars_info[2,0] = Params.nominal_speed + Params.speed_randomness*random.uniform(-1.0,1.0)
        else:
            self.numcars[1] += 1
            self.cars_info[0,0] = Params.start_environment
            self.cars_info[1,0] = Params.start_environment
            self.cars_info[2,0] = Params.nominal_speed + Params.speed_randomness*random.uniform(-1.0,1.0)
        
        print("Car0 on Lane " + str(self.cars_info2[0,0]))
        
        # Set the level of the ego agent
        self.cars_info2[5,0] = self.levk_distribution["ego"]
        
        #Update the number of vehicle that acts wrt this level 
        self.numcars_levk[self.cars_info2[5,0]] += 1
        
        # Assign level-k to each car randomly with respect to the given level-k distribution
        others = [*range(1,self.cars_info2.shape[1])]
        for i in range(self.levk_distribution["others"].size):
            if len(others) == 0:
                break
            
            if self.levk_distribution["others"][i] != 0:     
                num_levk = int(round(self.cars_info2.shape[1]*self.levk_distribution["others"][i]))
                if num_levk > len(others):
                    num_levk = len(others)
                    
                levk = random.sample(others,num_levk)
                self.cars_info2[5, levk] = i
                self.numcars_levk[i] = num_levk
                
                for car_j in levk:
                    others.remove(car_j)
        
        
        # Initialize the environment by placing vehicles around randomly
        force_end_init = False
        start_again = False
        total_trials = 0 #Number of trials to initialize the environment from scratch
        total_trial_limit = 3
        trial_limit = 5
        i = 1
        while i < self.cars_info.shape[1]:
            if start_again:
                start_again = False
                
            position_x = 0
            lane =  int(random.uniform(0,1)<self.bias_lane0vs1)#Choose a lane
            if self.numcars[0] >= self.maxcars_onlane0 and lane == 0:
                lane = 1;
            attempt_count = 0 # Number of attemps to place a single vehicle in a trial
            trials = 0 # Each trial ammounts to 250 attempts to place a single vehicle
            acceptable = False #Checks if the selected position is OK
            while not acceptable:
                if lane == 0:
                    position_x = (Params.end_merging_point - Params.far_distance) * random.uniform(0.0, 1.0)                
                else:
                    position_x = Params.add_car_point * random.uniform(0.0, 1.0) #Choose a position for cars on lane 1

                #Check if the minimum initial seperation between all cars is preserved
                for j in range(i):
                    if lane == self.cars_info2[0,j] and abs(position_x - self.cars_info[0,j]) < (
                            Params.min_initial_separation + Params.carlength):
                        acceptable = False
                        break
                    else:
                        acceptable = True
                attempt_count += 1

                #Choose a new lane and try again
                if attempt_count > 250 and not acceptable:
                    trials += 1;
                    # After 250 attempts, try the other lane
                    # If that lane is the ramp, then check if the limit is exceeded
                    if not(self.numcars[0] >= self.maxcars_onlane0 and lane == 1):
                        lane = (1 - lane)
                    attempt_count = 0
                                       
                # If the number of trials exceeds the limit, increment the total
                # number of trials, namely the number of vehicles for which the 
                # number of trials exceeds the limit
                if trials >= trial_limit and not acceptable:
                    total_trials += 1
                    if total_trials >= total_trial_limit:
                        # End the initilization with the last setting
                        self.cars_info = self.cars_info[:,0:i]
                        self.cars_info2 = self.cars_info2[:,0:i]
                        force_end_init = True
                        print("Force End Initialization at Car " + str(i))
                        print("Number of Cars: " + str(i))
                    else: # Start initilization from scratch
                        print("Trials >= "+str(trial_limit)+" for Car " + str(i))
                        i = 1 
                        start_again = True
                        self.numcars[0] = int(self.cars_info2[0,0]==0)
                        self.numcars[1] = int(self.cars_info2[0,0]==1)
                    break
            if not start_again and not force_end_init:
                self.cars_info[0,i] = position_x #Set the selected position
                self.cars_info2[0,i] = lane #Set the selected lane
                if lane == 0:
                    self.numcars[0] += 1
                    # If the vehicle is placed inside the merging region of the ramp,
                    # then assign velocity in a linear fashion
                    if position_x > Params.start_merging_point:
                        temp_vel = Params.nominal_speed*(0.5 + 0.5*(
                            Params.end_merging_point-position_x)/(
                                Params.end_merging_point-Params.start_merging_point)) + \
                                    Params.speed_randomness*random.uniform(-1.0,1.0)
                        self.cars_info[2,i] = temp_vel
                    else:
                        self.cars_info[2,i] = Params.nominal_speed +\
                            Params.speed_randomness*random.uniform(-1.0,1.0)
                else:
                    self.numcars[1] += 1
                    self.cars_info[2,i] = Params.nominal_speed + \
                        Params.speed_randomness*random.uniform(-1.0,1.0)
                i += 1

    # Add new car to the beginning of the environment
    def add_car(self,earliest):
        # Find possible level-k policies to be assigned according to given distribution
        possible_levs = np.where(self.levk_distribution["others"] != 0.0)[0]

        lev = 0
        if len(possible_levs) == 1:
            lev = possible_levs[0]
        else:
            prob = 0.0
            rnd = random.uniform(0.0,1.0)
            for i in possible_levs:
                prob += self.levk_distribution["others"][i]
                if prob > rnd:
                    lev = i
                    break
        
        position_x = 0
        lane = int(random.uniform(0,1)<self.bias_lane0vs1) #Choose a lane
        # If the number of vehicles on the ramp exceeds the limit, change the lane
        if self.numcars[0] >= self.maxcars_onlane0 and lane == 0:
            lane = 1;
            
        # Check if the closest vehicle in front is closer than the min initial separation
        # earliest = [ramp_pos, ramp_vel, main_road_pos, main_road_vel] 
        acceptable = earliest[0 + 2*lane] > (Params.min_initial_separation + Params.carlength)
        # Add car
        if acceptable:
            temp_info = np.zeros((self.cars_info.shape[0],self.cars_info.shape[1]+1))
            temp_info2 = np.zeros((self.cars_info2.shape[0],self.cars_info2.shape[1]+1),dtype=int)
            temp_info[:,:-1] = self.cars_info
            temp_info2[:,:-1] = self.cars_info2
            self.cars_info = temp_info
            self.cars_info2 = temp_info2
            del temp_info, temp_info2

            self.cars_info2[5,-1] = lev # Assign level-k
            self.cars_info[0,-1] = position_x #Set the selected position
            self.cars_info2[0,-1] = lane #Set the selected lane
            self.numcars[lane] += 1
            # Assign velocity with respect to the distance to the closest vehicle
            if earliest[2*lane] < (Params.far_distance + Params.carlength):
                self.cars_info[2,-1] = abs(earliest[1+2*lane] + \
                                           Params.speed_randomness*random.uniform(-1.0,1.0))
            else:    
                self.cars_info[2,-1] = Params.nominal_speed + \
                    Params.speed_randomness*random.uniform(-1.0,1.0)
                
        return acceptable

    def check_ego_state(self, nextstate=None, ignore_stopping=False):
        for idx in range(self.cars_info.shape[1]):
            if idx == 0:
                if self.cars_info2[0,0] == 0 and (self.cars_info[0,0] >= Params.end_merging_point):
                    return [True,0]
                continue

            if (abs(self.cars_info[0,idx]-self.cars_info[0,0]) <= (
                    Params.carlength + 0.1) and self.cars_info2[0,idx] == self.cars_info2[0,0]):
                if self.cars_info2[1,idx] == 5 and self.cars_info2[4,idx]:
                    return [True,1] # A car merged into ego
                elif self.cars_info2[1,0] == 5 or self.cars_info2[2,0] == 5:
                    return [True,2] #  Ego merged into a car
                elif (self.cars_info[1,idx]-self.cars_info[1,0] > Params.carlength):
                    return [True,3] # Ego crashed into a car in front
        
        if not ignore_stopping:
            fc_d = nextstate[2]*Params.max_sight_distance
            if self.cars_info[2,0] <= -Params.hard_decel_rate*Params.timestep:
                dist_end_merging = nextstate[-1]*Params.merging_region_length 
                if fc_d >= Params.far_distance:
                    return[True,6] # Ego stops even though there is enough space in front
                elif self.cars_info2[0,0] == 0 and 0 <= dist_end_merging <= Params.far_distance:
                    fl_d = nextstate[0]*Params.max_sight_distance
                    rl_d = nextstate[4]*Params.max_sight_distance
                    if fl_d >= Params.close_distance and rl_d <= -1.5*Params.far_distance:
                    # if fl_d >= Params.nominal_distance and rl_d <= -1.5*Params.far_distance:
                        return[True,7] # Ego doesn't merge even though there is enough space on the main lane
        
        return [False,-1]

    # Return the category of the ego vehicle state
    def translate_car0_state(self,car0_state):
        return Params.car0_states[car0_state]
    
    # Normalized the observation of a vehicle
    def normalize_state(self,state):
        msg = []
        msg.append(state[0]/Params.max_sight_distance) #fs_d
        msg.append(state[1]/(Params.max_speed-Params.min_speed)) #fs_v
        msg.append(state[2]/Params.max_sight_distance) #fc_d
        msg.append(state[3]/(Params.max_speed-Params.min_speed)) #fc_v
        msg.append(state[4]/Params.max_sight_distance) #rs_d
        msg.append(state[5]/(Params.max_speed-Params.min_speed)) #rs_v
        msg.append(state[6]/(Params.max_speed-Params.min_speed)) #vel
        msg.append(state[7]) #lane
        if state[8]>Params.merging_region_length: #dist_end_merging
            msg.append(1)
        elif state[8]<-Params.merging_region_length: #dist_end_merging
            msg.append(-1)
        else:    
            msg.append(state[8]/Params.merging_region_length) #dist_end_merging
        return msg

    #Find and return the observation of a car
    #Definition of variable names can be found in Message.py
    def get_Message(self, msgcar_id,normalize=False):
        fs_d = Params.max_sight_distance
        fs_v = Params.max_speed - self.cars_info[2,msgcar_id] + 0.1
        fc_d = Params.max_sight_distance
        fc_v = Params.max_speed - self.cars_info[2,msgcar_id] + 0.1        
        rs_d = -Params.max_sight_distance
        rs_v = Params.min_speed - self.cars_info[2,msgcar_id] - 0.1       

        car_fc = False # Is there a car in front
        car_fr = False # Is there a car on the right-front 
        for idx in range(self.cars_info.shape[1]):
            if idx == msgcar_id:
                continue

            # If the vehicle has passed throught the merging point on the ramp,
            # ignore it
            if (self.cars_info[0,idx] > Params.end_merging_point + Params.carlength and 
                self.cars_info2[0,idx] == 0):
                continue
            
            rel_position = self.cars_info[0,idx] - self.cars_info[0,msgcar_id]
            rel_velocity = self.cars_info[2,idx] - self.cars_info[2,msgcar_id]

            # The vehicle in on the left
            if self.cars_info2[0,idx] == (self.cars_info2[0,msgcar_id] + 1):
                # Behind the msg car
                if -Params.carlength > rel_position > rs_d:
                    rs_d = rel_position + Params.carlength
                    rs_v = rel_velocity
                # Next to the msg car
                elif 0 > rel_position > rs_d:
                    rs_d = 0
                    rs_v = rel_velocity
                # In front of the msg car
                elif Params.carlength <= rel_position < fs_d:
                    fs_d = rel_position - Params.carlength
                    fs_v = rel_velocity
                # Next to the msg car
                elif 0 <= rel_position < fs_d:
                    fs_d = 0
                    fs_v = rel_velocity
            # The vehicle in on the right
            elif self.cars_info2[0,idx] == (self.cars_info2[0,msgcar_id] - 1):
                # Behind the msg car
                if -Params.carlength > rel_position > rs_d:
                    rs_d = rel_position + Params.carlength
                    rs_v = rel_velocity
                # Next to the msg car
                elif 0 > rel_position > rs_d:
                    rs_d = 0
                    rs_v = rel_velocity
                # In front of the msg car
                elif Params.carlength <= rel_position < fs_d:
                    car_fr = True
                    fs_d = rel_position - Params.carlength
                    fs_v = rel_velocity
                # Next to the msg car
                elif 0 <= rel_position < fs_d:
                    car_fr = True
                    fs_d = 0
                    fs_v = rel_velocity
            # The vehicle in on the same lane
            elif self.cars_info2[0,idx] == self.cars_info2[0,msgcar_id]:
                # Safely distanced
                if Params.carlength <= rel_position < fc_d:
                    car_fc = True
                    fc_d = rel_position - Params.carlength
                    fc_v = rel_velocity
                # Crash
                elif 0 <= rel_position < fc_d:
                    car_fc = True
                    fc_d = 0
                    fc_v = rel_velocity
        
        # Consider the end of the merging region for fc and fs parameters
        if (self.cars_info2[0,msgcar_id] == 0) and ((
                Params.end_merging_point - Params.max_sight_distance) <= 
                self.cars_info[0,msgcar_id] <= Params.end_merging_point) and not car_fc: 
            fc_d = Params.end_merging_point - self.cars_info[0,msgcar_id]
            fc_v = -self.cars_info[2,msgcar_id]
        elif (self.cars_info2[0,msgcar_id] == 1) and ((
                Params.end_merging_point - Params.max_sight_distance) <= 
                self.cars_info[0,msgcar_id] <= Params.end_merging_point) and not car_fr: 
            fs_d = Params.end_merging_point - self.cars_info[0,msgcar_id]
            fs_v = -self.cars_info[2,msgcar_id] 
            
        # Provide zero values if the msg car passes the merging region on the main road
        if (self.cars_info2[0,msgcar_id] == 1 and 
            self.cars_info[0,msgcar_id] >= Params.end_merging_point):
            fs_d = 0
            fs_v = 0
                
        msg = [fs_d, fs_v, fc_d, fc_v, rs_d, rs_v,
               self.cars_info[2,msgcar_id], self.cars_info2[0,msgcar_id],
               Params.end_merging_point-self.cars_info[0,msgcar_id]]
        if normalize:
            msg = self.normalize_state(msg)
        return msg

        #Calculate and return the reward obtained
    def get_reward(self,crash,nextstate):
        performance = 1.0
        scale = 1.0
        
        wc = 1000 * scale # 1000 #   Collision
        wv = 5 * scale * performance # Velocity
        we = 5 * scale #*performance # Effort
        wh = 5 * scale # Headway
        wnm = 5 * scale #*performance # Not Merging
        ws = 60 * scale # 100  #  *performance # Velocity Less than 2.25m/s or Stopping on Lane-0 with dist_end_merging less than far distance
        # wm = 100
    
        c = 0
        if crash:
            c = -1
            
        # Velocity
        v_coeff = 1 # 0.2
        if self.cars_info[2,0] > Params.nominal_speed:
            v = v_coeff*(self.cars_info[2,0] - Params.nominal_speed)/(Params.max_speed - Params.nominal_speed)
        else:
            v = (self.cars_info[2,0] - Params.nominal_speed)/(Params.nominal_speed-Params.min_speed)
        
        if self.cars_info[2,0] > (Params.nominal_speed+Params.min_speed)/2:
            if self.cars_info2[3,0] == 0:
                e = 0
            elif self.cars_info2[3,0] == 1:
                e = -0.25
            elif self.cars_info2[3,0] == 2:
                e = -1
            else:
                e = 0
        else:
            e = 0
        
        # Headway
        h = 0
        fc_d = nextstate[2]*Params.max_sight_distance
        if fc_d <= Params.close_distance:
            h = -1
        elif fc_d <= Params.nominal_distance:
            h = 1*(fc_d-Params.nominal_distance)/(Params.nominal_distance-Params.close_distance)
        elif self.cars_info[2,0] > -Params.hard_decel_rate*Params.timestep:
            if fc_d <= Params.far_distance:
                h = 1*(fc_d - Params.nominal_distance)/(Params.far_distance-Params.nominal_distance)
            elif fc_d > Params.far_distance:
                h = 1
        
        # Not Merging
        nm = 0
        if self.cars_info2[0,0] == 0:  # Staying on Lane-0 
            nm = -1 

        # Stopping
        # s = 0
        # if self.cars_info[2,0] <= -Params.hard_decel_rate*Params.timestep:
        #     dist_end_merging = nextstate[-1]*Params.merging_region_length 
        #     if fc_d >= Params.far_distance:
        #         s = -1
        #     elif self.cars_info2[0,0] == 0 and 0 <= dist_end_merging <= Params.far_distance:
        #         # s = -0.05
        #         fl_d = nextstate[0]*Params.max_sight_distance
        #         rl_d = nextstate[4]*Params.max_sight_distance
        #         if fl_d >= Params.close_distance and rl_d <= -1.5*Params.far_distance:
        #         # if fl_d >= Params.nominal_distance and rl_d <= -1.5*Params.far_distance:
        #             s = -1
        #         else:
        #             s = -0.05
            
        # New stopping parameter code for level k
        fl_d = nextstate[0]*Params.max_sight_distance
        rl_d = nextstate[4]*Params.max_sight_distance
        dist_end_merging = nextstate[-1]*Params.merging_region_length 
        s = 0
        if self.cars_info[2,0] <= -Params.hard_decel_rate*Params.timestep: # velocity is small enough
            if self.cars_info2[0,0] == 1: # Main Road Case
                # if (self.cars_info[2,0] <= -Params.hard_decel_rate*Params.timestep or self.cars_info2[1,0] != 4) and fc_d >= Params.far_distance and (dist_end_merging >= Params.far_distance or dist_end_merging < 0):
                if (self.cars_info2[1,0] not in (1, 3)) and fc_d >= Params.far_distance and (dist_end_merging >= Params.far_distance or dist_end_merging < 0):
                    s = -1  # Penalize if the vehicle is too far behind and not accelerating
                    #print("a")
                else:
                    s = 0  # No penalty otherwise
                    #print("b")
            elif self.cars_info2[0,0] == 0: # ramp
                if (dist_end_merging < Params.merging_region_length): # inside merging region
                    if self.cars_info2[1,0] != 5:
                        if fl_d >= Params.close_distance and abs(rl_d) >= (1.5 * Params.far_distance):
                            s = -1  # Penalize if the ego vehicle is not merging while it is safe to
                            # print("c")
                            # TODO almost like nm parameter, is this ok?
                        elif dist_end_merging <= Params.far_distance:
                            s = -0.05  # Small penalty for stopping close to the end of the merging zone. it may not be safe to merge.
                            # print("d")
                    else:
                        s = 0  # No penalty otherwise
                        # print("e")
                else: # not yet in the merging region
                    if fc_d >= Params.far_distance: # and it is slow despite the space in front
                        s = -0.20
                        # print("f")

        return wc*c + wv*v + we*e + wh*h + wnm*nm + ws*s #+ wm*m


    # Determines the action of a level-0 car
    def get_level0_action(self, msg, car_id):
        hard_decel_TTC = 3 #3 # Time-to-Collision for hard deceleration action
        decel_TTC = 7 #5 # Time-to-Collision for deceleration action
        
        act = 0
        if self.cars_info2[0,car_id] == 0:
            # Inside merging region
            if (msg.dist_end_merging < Params.merging_region_length - Params.carlength):
                
                # Threshold for merging decision
                rand_merging_cond = ((Params.merging_region_length - msg.dist_end_merging)
                                     /Params.merging_region_length)**2
                
                # Merge if the distance to the merging point is safe and
                # a randomly sampled value is greater than the threshold
                if msg.dist_end_merging < Params.far_distance or random.uniform(0.0, 1.0) < rand_merging_cond:
                    
                    # Assign a non-zero value for fs_v to calculate TTC to
                    # the car on the side-front
                    fs_v = (msg.fs_v > 0)*0.01 - (msg.fs_v <= 0)*min(msg.fs_v,-0.01)
                    
                    # Check TTC and the distance to the vehicle on the side-front
                    if (msg.fs_d/fs_v >= hard_decel_TTC and msg.fs_d > Params.close_distance) or (
                        msg.fs_d > Params.far_distance):
                        
                        # Assign a non-zero value for rs_v to calculate TTC to
                        # the car on the side-rear                
                        rs_v = (msg.rs_v < 0)*-0.01 + (msg.rs_v >= 0)*max(msg.rs_v,0.01)
                        
                        # Check TTC and the distance to the vehicle on the side-front
                        if (-msg.rs_d/rs_v >= hard_decel_TTC and -msg.rs_d > Params.close_distance) or (
                            -msg.rs_d > 1.0*Params.far_distance):
                            act = 5 # Merge
                            
            # If the decision is not to merge, then check for others                
            if act == 0:
                if (-msg.fc_d/min(msg.fc_v,-0.01) <= hard_decel_TTC and msg.fc_d > Params.close_distance) or (
                        msg.fc_d <= Params.close_distance):
                    act = 4 # Hard-Decelerate
                elif (-msg.fc_d/min(msg.fc_v,-0.01) <= decel_TTC and msg.fc_d > Params.close_distance) or (
                        self.cars_info[0,car_id] > Params.start_merging_point and (
                            msg.velocity > Params.nominal_speed*msg.dist_end_merging/Params.merging_region_length)):                 
                    act = 2 # Decelerate
                elif msg.dist_end_merging >= Params.far_distance and (
                        msg.fc_d > Params.close_distance) and msg.fc_v > 0.01 and msg.velocity < Params.nominal_speed:
                    act = 1 # Accelerate
        else:
            if (-msg.fc_d/min(msg.fc_v,-0.01) <= hard_decel_TTC and msg.fc_d > Params.close_distance) or (
                    msg.fc_d <= Params.close_distance):
                act = 4 # Hard-Decelerate
            elif (-msg.fc_d/min(msg.fc_v,-0.01) <= decel_TTC and msg.fc_d > Params.close_distance):
                act = 2 # Decelerate
            elif msg.fc_d > Params.close_distance and msg.fc_v > 0.01 and (
                    msg.velocity < Params.nominal_speed or Params.end_merging_point < self.cars_info[0,car_id]):
                act = 1 # Accelerate
                
        return act
    
    #This checks the selected action and attains appropriote effort value to the selected action
    def set_action(self, act, car_id):
        self.cars_info2[4,car_id] = False
        if act == 0:
            self.cars_info2[3,car_id] = 0
        else:
            self.cars_info2[3,car_id] = 1
            if 3 <= act <= 4: # hard_accel and hard_decel
                self.cars_info2[3,car_id] = 2
            elif 5 == act: # Merge
                self.cars_info2[3,car_id] = 3

        self.cars_info2[2,car_id] = self.cars_info2[1,car_id]
        self.cars_info2[1,car_id] = act

	#Position update code of trained level-k policies
    def update_motion(self, car_id, act=None, msg=None, get_acc=False):
        # If k=0, then get level-0 action
        if self.cars_info2[5,car_id] == 0:
            act = self.get_level0_action(msg,car_id)

        # Set action
        self.set_action(act, car_id)

        prev_vel = self.cars_info[2,car_id] # Previous velocity
        acc = 0 
        epsilon = 0.01
        
        #Change the lane if it is intended
        signvy = 0
        if self.cars_info2[1,car_id] == 5:
            signvy = 1
            self.cars_info2[0,car_id] += signvy
            self.numcars[1] += signvy
            self.numcars[0] -= signvy
            self.cars_info2[4,car_id] = True

        #Implements accelerate action by sampling an acceleration
        elif self.cars_info2[1,car_id] == 1 and self.cars_info[2,car_id] < (Params.max_speed - epsilon):
            acc = min(0.25 + np.random.exponential(scale=0.75),Params.actions[self.cars_info2[1,car_id]][1])
            # acc = (random.uniform(0.0,Params.actions[self.cars_info2[1,car_id]][1])+0.5)
            self.cars_info[2,car_id] += acc * Params.timestep

        #Implements decelerate action by sampling an acceleration
        elif self.cars_info2[1,car_id] == 2 and self.cars_info[2,car_id]> (Params.min_speed + epsilon):
            acc = max(-0.25 - np.random.exponential(scale=0.75),Params.actions[self.cars_info2[1,car_id]][1])
            # acc = (random.uniform(0.0,Params.actions[self.cars_info2[1,car_id]][1])-0.5)
            self.cars_info[2,car_id] +=  acc * Params.timestep

        #Implements hard accelerate action by sampling an acceleration
        elif self.cars_info2[1,car_id] == 3 and self.cars_info[2,car_id]< (Params.max_speed - epsilon):
            acc = min(2 + np.random.exponential(scale=0.75),Params.actions[self.cars_info2[1,car_id]][1])
            # acc = (-abs(0.3 * random.gauss(0, 1)) + Params.actions[self.cars_info2[1,car_id]][1]) 
            self.cars_info[2,car_id] += acc * Params.timestep

        #Implements hard decelerate action by sampling an acceleration
        elif self.cars_info2[1,car_id] == 4 and self.cars_info[2,car_id] > (Params.min_speed + epsilon):
            acc = max(-2 - np.random.exponential(scale=0.75),Params.actions[self.cars_info2[1,car_id]][1])
            # acc = (abs(0.3 * random.gauss(0, 1)) + Params.actions[self.cars_info2[1,car_id]][1])
            self.cars_info[2,car_id] += acc * Params.timestep

        #Implements maintain action by sampling an acceleration value
        elif self.cars_info2[1,car_id] == 0 and (self.cars_info[2,car_id] < (Params.max_speed - epsilon)) and (
            self.cars_info[2,car_id] > (Params.min_speed + epsilon)):
            acc = np.random.laplace(scale=0.1)
            acc = (acc>=0)*min(acc,0.25)+(acc<0)*max(acc,-0.25)
            # acc = random.gauss(0.0, 0.075)
            self.cars_info[2,car_id] += acc * Params.timestep

        #Assign the velocity within limits
        if self.cars_info[2,car_id] < Params.min_speed:
            acc = (prev_vel-Params.min_speed)/Params.timestep
            self.cars_info[2,car_id] = Params.min_speed
        elif self.cars_info[2,car_id] > Params.max_speed:
            acc = (Params.max_speed-prev_vel)/Params.timestep
            self.cars_info[2,car_id] = Params.max_speed
      
        # Update acceleration
        self.cars_info[3,car_id] = acc
        
        # Update previous position
        self.cars_info[1,car_id] = self.cars_info[0,car_id]

        # Update current position
        if self.cars_info[2,car_id] == Params.min_speed:
            self.cars_info[0,car_id]+= (prev_vel+0.5*(Params.min_speed-prev_vel))*Params.timestep
        elif self.cars_info[2,car_id] == Params.max_speed:
            self.cars_info[0,car_id]+= (prev_vel+0.5*(Params.max_speed-prev_vel))*Params.timestep
        else:
            self.cars_info[0,car_id] += (prev_vel+0.5*(acc*Params.timestep))*Params.timestep

        reached_end = False
        # Check if the car has passed its corresponding end point
        if car_id == 0:
            reached_end = self.cars_info[0,car_id] > Params.end_for_car0
        else:
            reached_end = self.cars_info[0,car_id] > Params.add_car_point
        if get_acc:
            return reached_end, acc
        return reached_end
	
    # Check positions of each car, if a crash occurred, update the tail's position
    # as the rear-end of the car in front and velocity as the car in front
    def check_reset_positions(self):
        # Get the orderred indices of vehicles with respect to their positions
        orderred_cars = np.argsort(self.cars_info[0,:])
        
        lane0 = []
        lane1 = []
        
        # Collect orderred vehicles into ramp and main road lists separately
        for i in range(self.cars_info.shape[1]):
            if self.cars_info2[0,orderred_cars[i]] == 0:
                # If the vehicle passed the merging point, reverse it back
                # in order to prevent unrealistic position
                if orderred_cars[i] != 0 and self.cars_info[0,orderred_cars[i]] > Params.end_merging_point:
                    self.cars_info[0,orderred_cars[i]] = Params.end_merging_point
                    self.cars_info[2,orderred_cars[i]] = 0
                lane0.append(orderred_cars[i])
            else:
                lane1.append(orderred_cars[i])
        
        # Reset positions on the ramp if a collision occurs
        for i in reversed(range(len(lane0)-1)):
            current_id = lane0[i]
            current_leader_id = lane0[i+1]
            if (current_id != 0 and 
                0 <= self.cars_info[0,current_leader_id] - self.cars_info[0,current_id] < Params.carlength):
                if self.cars_info2[4,current_id] == False:
                    self.cars_info[0,current_id] = self.cars_info[0,current_leader_id] - Params.carlength
                    self.cars_info[2,current_id] = self.cars_info[2,current_leader_id]

        # Reset positions on the main road if a collision occurs                    
        for i in reversed(range(len(lane1)-1)):
            current_id = lane1[i]
            current_leader_id = lane1[i+1]
            if (current_id != 0 and 
                0 <= self.cars_info[0,current_leader_id] - self.cars_info[0,current_id] < Params.carlength):
                if self.cars_info2[4,current_id] == False:
                    self.cars_info[0,current_id] = self.cars_info[0,current_leader_id] - Params.carlength
                    self.cars_info[2,current_id] = self.cars_info[2,current_leader_id]             