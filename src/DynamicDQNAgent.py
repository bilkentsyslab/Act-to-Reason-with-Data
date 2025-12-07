#This file implements DQN for Dynamic Agent
# © 2020 Cevahir Köprülü All Rights Reserved
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Activation
from keras.optimizers import Adam
from collections import deque
from src.DQNAgent import DQNAgent
import pickle
import random
import numpy as np
from scipy.special import softmax
from keras.callbacks import LearningRateScheduler #to change the learning rate dynamically
import gc


# import tensorflow as tf
# tf.set_random_seed(0)
# random.seed(0)
# np.random.seed(0)

class DynamicDQNAgent:
    def __init__(self, dynamic_state_size, dynamic_action_size, state_size, action_size, levelk_config):
        self.dynamic_state_size = dynamic_state_size
        self.dynamic_action_size = dynamic_action_size
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000) #experience set, i.e. memory
        self.gamma = 0.95    # discount rate
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.T_SCALE = 1.0
        self.MIN_T = self.T_SCALE * 1.0
        self.MAX_T = self.T_SCALE * np.float64(50)
        self.T = self.T_SCALE * np.float64(50) #Initial boltzmann temperature
        self.boltzmann_sampling = True # Boltzmann Sampling
        self.dynamic_driving_boltzmann_sampling = levelk_config["boltzmann_sampling"]
        self.random_dynamic_strategy = False # Dynamic agent randomly selects a level-k strategy
        self.agentlevk = {}


        #new version: /to avoid unable to load model error
        for i in range (3):
            if levelk_config["paths"][i+1] != None:
                self.agentlevk[i] = self._load_levelk(levelk_config["paths"][i+1], self.dynamic_driving_boltzmann_sampling)

        #old version:
        #self.agentlevk[0] = self._load_levelk(levelk_config["paths"][1],self.dynamic_driving_boltzmann_sampling)
        #self.agentlevk[1] = self._load_levelk(levelk_config["paths"][2],self.dynamic_driving_boltzmann_sampling)
        #self.agentlevk[2] = self._load_levelk(levelk_config["paths"][3],self.dynamic_driving_boltzmann_sampling)


    #This function returns the NN model for DQN
    #Weight Initializer: Glorot Uniform
    def _build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(1,self.dynamic_state_size))) #number of observations
        model.add(Dense(256)) # Hidden Layer 1
        model.add(Activation('relu'))
        model.add(Dense(256)) # Hidden Layer 2
        model.add(Activation('relu'))
        model.add(Dense(128)) # Hidden Layer 3
        model.add(Activation('relu'))
        model.add(Dense(self.dynamic_action_size)) # Output Layer
        model.add(Activation('linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    #Loads level-k agent model in the given path
    def _load_levelk(self,path,random):
        agentlevk = DQNAgent(self.state_size, self.action_size) #Level-k Agent
        
        agentlevk.load(path) #Loads the model of trained level-k agent
        agentlevk.T = 1 #Sets the boltzmann temp. of Level-k cars to 1, prevents extra-random actions #BURASI
        agentlevk.boltzmann_sampling = True # random
        return agentlevk
    
    #Adds given experience to the experience set
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    #Updates the target model by making its weights equal to the original model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_average_layer_weights(self):
        ave_weight = []
        ave_bias = []
        for idx_layer, layer in enumerate(self.model.layers):
            if idx_layer % 2 == 0:
                continue
            ave_weight.append(layer.get_weights()[0].mean())
            ave_bias.append(layer.get_weights()[1].mean())

        return ave_weight, ave_bias
    
    #Chooses a level-k strategy. Return an action of choice 5-6
    def act(self, state, remove_merging=False, get_qvals=False):
        if self.random_dynamic_strategy:
            random_strategy = random.randint(0, 2)

            while ( random_strategy not in self.agentlevk):
                random_strategy = random.randint(0, 2)

 
            if get_qvals:
                
                levk_qvals = []
                for i in range(3):
                    try:
                        levk_qvals.append(self.agentlevk[i].act(state=state[:, :, 0:self.state_size], remove_merging=remove_merging, get_qvals=get_qvals))
                    except KeyError:
                        print(f"Agent {i+1} does not exist in self.agentlevk")
                        
                return [["Random-Strategy "+ str(random_strategy), random_strategy], levk_qvals]
            else:
                return [["Random-Strategy "+ str(random_strategy), random_strategy], self.agentlevk[random_strategy].act(
                    state=state[:, :, 0:self.state_size], remove_merging=remove_merging)]
        else:
            q_values = self.model(state)[0].numpy()
            if remove_merging:
                act_values = q_values[:5]   #for 5 different actions wrong, it should be 3, if is redundant 
            else:
                act_values = q_values

            greedy = np.argmax(act_values)
            exp_values = softmax(act_values / self.T)
            rand = np.random.rand()

            if get_qvals:  # Return Q values with action-related information

                levk_qvals = []
                for i in range(3):
                    try:
                        levk_qvals.append(self.agentlevk[i].act(state=state[:, :, 0:self.state_size], remove_merging=remove_merging, get_qvals=get_qvals))
                    except KeyError:
                        print(f"Agent {i+1} does not exist in self.agentlevk")
                """old version:
                levk_qvals = [
                    self.agentlevk[0].act(state=state[:, :, 0:self.state_size], remove_merging=remove_merging, get_qvals=get_qvals),
                    self.agentlevk[1].act(state=state[:, :, 0:self.state_size], remove_merging=remove_merging, get_qvals=get_qvals),
                    self.agentlevk[2].act(state=state[:, :, 0:self.state_size], remove_merging=remove_merging, get_qvals=get_qvals)
                ]

                """
                # If softmax outputs are not finite, return the greedy action
                if not np.all(np.isfinite(exp_values)):
                    return [["Argmax-Inf", q_values.tolist(), greedy], levk_qvals]

                # Calculate cumulative probabilities
                cumulative_prob = np.cumsum(exp_values)

                # Boltzmann sampling
                for i, cum_prob in enumerate(cumulative_prob):
                    if self.boltzmann_sampling and (rand < cum_prob):
                        if greedy == i:
                            return [["Random-Argmax", q_values.tolist(), greedy], levk_qvals]
                        else:
                            return [["Random", q_values.tolist(), i], levk_qvals]
                return [["Argmax", q_values.tolist(), greedy], levk_qvals]
            else:  # Return only action-related information
                # If softmax outputs are not finite, return the greedy action
                if not np.all(np.isfinite(exp_values)):
                    return [["Argmax-Inf", greedy], self.agentlevk[greedy].act(
                        state=state[:, :, 0:self.state_size], remove_merging=remove_merging)]

                # Calculate cumulative probabilities
                cumulative_prob = np.cumsum(exp_values)

                # Boltzmann sampling
                for i, cum_prob in enumerate(cumulative_prob):
                    if self.boltzmann_sampling and (rand < cum_prob):
                        if greedy == i:
                            return [["Random-Argmax", greedy], self.agentlevk[greedy].act(
                                state=state[:, :, 0:self.state_size], remove_merging=remove_merging)]
                        else:
                            return [["Random", i], self.agentlevk[i].act(
                                state=state[:, :, 0:self.state_size], remove_merging=remove_merging)]
                # If none of the above, return the greedy action
                return [["Argmax", greedy], self.agentlevk[greedy].act(
                    state=state[:, :, 0:self.state_size], remove_merging=remove_merging)]

         
    # Returns actions and level-k strategies of multiple dynamic agents 
    # through Boltzmann Sampling
    # remove_merging parameter determines whether to
    # remove the "merge" action from the last layer
    def act_inbatch(self, states, remove_merging=False):
        if self.random_dynamic_strategy:
            acts = np.zeros((states.shape[0]))
            random_strategies = np.array(random.choices([0, 1, 2], k=states.shape[0]))      #generates states many random number array of 0,1,2 elements
            l1 = np.where(random_strategies == 0)[0]
            l2 = np.where(random_strategies == 1)[0]
            l3 = np.where(random_strategies == 2)[0]  #finds the set of indices where random+strategy is i for li.

            
            if l1.size != 0:
                acts[l1] = self.agentlevk[0].act_inbatch(states[l1, :, 0:self.state_size], remove_merging)
            if l2.size != 0:
                acts[l2] = self.agentlevk[1].act_inbatch(states[l2, :, 0:self.state_size], remove_merging)
            if l3.size != 0:
                acts[l3] = self.agentlevk[2].act_inbatch(states[l3, :, 0:self.state_size], remove_merging)
            
            return acts, random_strategies

        else:
            dataset = tf.data.Dataset.from_tensor_slices(states)
            dataset = dataset.batch(states.shape[0])

            q_values_batch = []
            for batch in dataset:
                q_values_batch.append(self.model(batch, training=False).numpy())
            
            q_values_batch = np.vstack(q_values_batch)

            if remove_merging:
                q_values_batch = q_values_batch[:, :5]

            exp_values_batch = softmax(q_values_batch / self.T, axis=1)

            select_levk = np.zeros((q_values_batch.shape[0]))
            acts = np.zeros((q_values_batch.shape[0]))

            if self.boltzmann_sampling:
                rand_batch = np.random.rand(states.shape[0], 1)
                cumulative_prob_batch = np.cumsum(exp_values_batch, axis=1)
                cumulative_prob_batch = cumulative_prob_batch / cumulative_prob_batch[:, -1, np.newaxis]

                not_nan_indices = np.where(np.all(np.isfinite(exp_values_batch), axis=1))[0]
                nan_indices = np.where(~np.all(np.isfinite(exp_values_batch), axis=1))[0]

                if not_nan_indices.size > 0:
                    exp_values_batch_not_nan = exp_values_batch[not_nan_indices]
                    cumulative_prob_batch_not_nan = cumulative_prob_batch[not_nan_indices]
                    rand_batch_not_nan = rand_batch[not_nan_indices]

                    actions_not_nan = np.argmax(cumulative_prob_batch_not_nan > rand_batch_not_nan, axis=1)
                    select_levk[not_nan_indices] = actions_not_nan

                if nan_indices.size > 0:
                    select_levk[nan_indices] = np.argmax(q_values_batch[nan_indices], axis=1)

                l1 = np.where(select_levk == 0)[0]
                l2 = np.where(select_levk == 1)[0]
                l3 = np.where(select_levk == 2)[0]

                if l1.size != 0:
                    acts[l1] = self.agentlevk[0].act_inbatch(states[l1, :, 0:self.state_size], remove_merging)
                if l2.size != 0:
                    acts[l2] = self.agentlevk[1].act_inbatch(states[l2, :, 0:self.state_size], remove_merging)
                if l3.size != 0:
                    acts[l3] = self.agentlevk[2].act_inbatch(states[l3, :, 0:self.state_size], remove_merging)

                return acts, select_levk
            else:
                select_levk = np.argmax(q_values_batch, axis=1)
                l1 = np.where(select_levk == 0)[0]
                l2 = np.where(select_levk == 1)[0]
                l3 = np.where(select_levk == 2)[0]

                if l1.size != 0:
                    acts[l1] = self.agentlevk[0].act_inbatch(states[l1, :, 0:self.state_size], remove_merging)
                if l2.size != 0:
                    acts[l2] = self.agentlevk[1].act_inbatch(states[l2, :, 0:self.state_size], remove_merging)
                if l3.size != 0:
                    acts[l3] = self.agentlevk[2].act_inbatch(states[l3, :, 0:self.state_size], remove_merging)

                return acts, select_levk

    def compute_q_loss(self, predicted_q_values, target_q_values):
        return np.mean((predicted_q_values - target_q_values) ** 2)

    #Experience replay
    def replay(self, batch_size):
        batch_size = min(batch_size, len(self.memory))
        minibatch = np.array(random.sample(self.memory, batch_size), dtype="object")
        
        # Extract states and next states
        X = np.squeeze(np.stack(minibatch[:,0]), axis=1)  # States
        next_states = np.squeeze(np.stack(minibatch[:,3]), axis=1)  # Next states
        
        # Directly calling the model to predict current and next Q values
        Y = self.model(X, training=False).numpy()  # Current Q-values
        next_Q_values = self.target_model(next_states, training=False).numpy()  # Next Q-values using target model

        # Rewards from past experiences
        R = np.reshape(np.stack(minibatch[:,2]), [batch_size, 1])
        
        # Calculate the max Q value for future states from the target model
        Q_val = np.max(next_Q_values, axis=1, keepdims=True)
        
        # Information about whether the episode has ended
        done = np.reshape(np.stack(minibatch[:,-1]), [batch_size, 1])

        # Update the targets for training
        actions = minibatch[:, 1].astype(int)
        updates = R + self.gamma * Q_val * (1 - done)
        Y[np.arange(batch_size), actions] = updates.squeeze()

        # Fit the model with the updated targets
        #if ((state_no+10) %20 == 0  and state_no != 10 and state_no):
        #    callback = tf.keras.callbacks.LearningRateScheduler(self.scheduler1, verbose= 1)
        #else:
        #callback = tf.keras.callbacks.LearningRateScheduler(self.scheduler1, verbose= 1)

        #history = self.model.fit(X, Y, batch_size=batch_size, epochs=1, callbacks=[callback],  verbose=0)
        
        history = self.model.fit(X, Y, batch_size=batch_size, epochs=1,  verbose=0)

        return history.history['loss']
    
    

    def scheduler1 (self, epoch, lr):

        lr = self.learning_rate
        return lr

    def scheduler (self, epoch, lr):

        if  lr < 0.0005:
           return lr
        else :
          return lr*0.999997

    def update_temperature(self, step):
        self.T = np.maximum(self.T - (self.MAX_T-self.MIN_T)*step, self.MIN_T)
        print(self.T)

    def load_memory(self, fname):
        with open(fname, "rb") as input_file:
            self.memory = pickle.load(input_file)
    
    def save_memory(self,fname, episode_no, run_no):
        with open(fname , "wb") as output_file:
            pickle.dump(self.memory, output_file)
            
    #Loads the Boltzmann temperature and total timesteps
    def load_config(self, config_fname):
        with open(config_fname, "rb") as input_file:
            # [self.T,self.epsilon, total_timesteps] = pickle.load(input_file)
            [self.T, total_timesteps] = pickle.load(input_file)
        return total_timesteps
    
    #Saves the last Boltzmann temperature and total timesteps
    def save_config(self, total_timesteps, config_fname, episode_no, run_no):
        with open(config_fname ,  "wb") as output_file:
            # pickle.dump([self.T,self.epsilon, total_timesteps], output_file)
            pickle.dump([self.T, total_timesteps], output_file)
            
    #Loads the model, and the target weights for retraining
    def load(self, fname, tfname="", retrain=False):
        self.model = load_model(fname)
        if retrain:
            self.target_model.load_weights(tfname) 
            
    #Saves the model, and the target weights for retraining
    def save(self, fname,  tfname="",  backup = False):
        self.model.save(fname)
        if backup:
            self.target_model.save_weights(tfname)
