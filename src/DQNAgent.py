#This file implements DQN for level-k agents
# © 2020 Cevahir Köprülü All Rights Reserved

import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Activation
from keras.callbacks import LearningRateScheduler #to change the learning rate dynamically
from keras.optimizers import Adam
import pickle
from scipy.special import softmax
import math
#import gc

# import tensorflow as tf
# tf.set_random_seed(0)
# random.seed(0)
# np.random.seed(0)
class DQNAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size #9-dim
        self.action_size = action_size #5?
        self.memory = deque(maxlen=50000) #experience set, i.e. memory
        self.gamma = np.float32(0.95)    # discount rate
        self.learning_rate = np.float32(0.002)
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.T_SCALE = 1.0
        self.MIN_T = self.T_SCALE * np.float32(1.0) #1 where boltzman effect stops
        self.MAX_T = self.T_SCALE * np.float32(50)
        self.T = self.T_SCALE * np.float32(50) #Initial boltzmann temperature, same with max
        self.boltzmann_sampling = True # Boltzmann Sampling


        
    #This function returns the NN model for DQN
    #Weight Initializer: Glorot Uniform


   
    def _build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(1,self.state_size))) #number of observations
        model.add(Dense(256)) # Hidden Layer 1
        model.add(Activation('relu'))
        model.add(Dense(256)) # Hidden Layer 2
        model.add(Activation('relu'))
        model.add(Dense(128)) # Hidden Layer 3
        model.add(Activation('relu'))
        model.add(Dense(self.action_size)) # Output Layer
        model.add(Activation('linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

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
    
    #Chooses an action
    def act(self, state, remove_merging=False, get_qvals=False):
        q_values = self.model(state)[0].numpy()
        if remove_merging:
            act_values = q_values[:5]
        else:
            act_values = q_values
            
        greedy = np.argmax(act_values)
        exp_values = softmax(act_values / self.T)
        rand = np.random.rand()
                   
        if get_qvals: # Return Q values with action related information
            # If softmax outputs are zero or infinite valued,
            # return the greedy action                
            if not np.all(np.isfinite(exp_values)):
                return ["Argmax-Inf", q_values.tolist(), greedy]
            
            # Since exp_values are probabilities, no need to divide by sum
            cumulative_prob = np.cumsum(exp_values) #this adds the probabilities and puts into an array of size exp_values
            
            # Take random action by Boltzmann Sampling if the random number passes
            # the cumulative probability
            for i, cum_prob in enumerate(cumulative_prob):  
                if self.boltzmann_sampling and (rand < cum_prob):
                    if greedy == i:
                        return ["Random-Argmax",q_values.tolist(), greedy]
                    else:
                        return ["Random",q_values.tolist(), i]
            return ["Argmax", q_values.tolist(),greedy]
        else: # Return only action r    elated information
            # If softmax outputs are zero or infinite valued,
            # return the greedy action               
            if not np.all(np.isfinite(exp_values)):
                return ["Argmax-Inf", greedy]
            
            # Since exp_values are probabilities, no need to divide by sum
            cumulative_prob = np.cumsum(exp_values)
            
            # Take random action by Boltzmann Sampling if the random number passes
            # the cumulative probability
            for i, cum_prob in enumerate(cumulative_prob):
                if self.boltzmann_sampling and (rand < cum_prob):
                    if greedy == i:
                        return ["Random-Argmax", greedy]
                    else:
                        return ["Random", i]
            # If none of the above, return the greedy action
            return ["Argmax", greedy]

        
    # Returns actions of multiple agents that are selected via greedy policy
    # No forced decel/hard_decel is considered
    # Returns actions of multiple agents that are selected via greedy policy
    def act_inbatch(self, states, remove_merging=False):
        
        dataset = tf.data.Dataset.from_tensor_slices(states)
        dataset = dataset.batch(states.shape[0])
        
        q_values_batch = []
        for batch in dataset:
            q_values_batch.append(self.model(batch, training = False).numpy()) #predictions for (q vals,,) for sample batch states etc.

        q_values_batch = np.vstack(q_values_batch)
        
        # Ignore "merge" action unit if specified
        if remove_merging:
            q_values_batch = q_values_batch[:, :5]
        
        # Apply softmax to compute the probabilities for Boltzmann sampling
        exp_values_batch = softmax(q_values_batch / self.T, axis=1)
        
        if self.boltzmann_sampling:
            # Generate random numbers for each sample
            rand_batch = np.random.rand(states.shape[0], 1)
            cumulative_prob_batch = np.cumsum(exp_values_batch, axis=1)
            cumulative_prob_batch = cumulative_prob_batch / cumulative_prob_batch[: , -1, np.newaxis]
            # Iterate over each set of cumulative probabilities to determine actions
            actions = np.argmax(cumulative_prob_batch > rand_batch, axis = 1)

        else:
            actions = np.argmax( q_values_batch  , axis = 1)

        return actions
   

    #Experience replay function, which is explained in the original DQN Paper
    def replay(self, batch_size):
        batch_size = min(batch_size, len(self.memory))
        minibatch = np.array(random.sample(self.memory, batch_size), dtype="object")
        
        X = np.squeeze(np.stack(minibatch[:,0]),axis=1)
        next_states = np.squeeze(np.stack(minibatch[: , 3]), axis=1)

        Y = self.model(X, training = False).numpy()
        next_Q_Values = self.target_model(next_states, training = False).numpy()


        # Rewards of past experiences
        R = np.reshape(np.stack(minibatch[:,2]),[batch_size,1])
        
        # Q values of past experiences obtained from the target model
        Q_val = np.max(next_Q_Values,axis=1, keepdims = True)
        
        # Information about the status of the car, namely whether
        # the car passed the final point or crashed
        done = np.reshape(np.stack(minibatch[:,-1]),[batch_size,1])

        actions = minibatch[: , 1].astype(int)
        updates = R + self.gamma * Q_val * (1-done)
        Y[np.arange(batch_size), actions] = updates.squeeze()

        
        callback = tf.keras.callbacks.LearningRateScheduler(self.scheduler, verbose= 0)
        history = self.model.fit(X, Y, batch_size=batch_size, epochs=1, callbacks=[callback], verbose= 0) 
        #garbage collector for Syslab computer for Ram memory to be disposed (suppresses GPU)

        #gc.collect()
        #tf.keras.backend.clear_session()
        #upper code segment was aimed to avoid memory leaks. It achieves less RAM usage. Yet, also leads to drop in GPU use!



        #with open("new_learning_rates.txt", "a") as file:   #grows a lot
        #     file.write("New learning rate became: " + str(self.model.optimizer.lr))
        return history.history['loss']
    


    def scheduler (self, epoch, lr):

        if  lr < 0.001:
           return lr
        else:
          return lr*0.999997

    def update_temperature(self, step):
        self.T = np.maximum(self.T - (self.MAX_T-self.MIN_T)*step, self.MIN_T) 
        
    def load_memory(self, fname):
        with open(fname, "rb") as input_file:
            self.memory = pickle.load(input_file)
            
    def save_memory(self,fname):
        with open(fname, "wb") as output_file:
            pickle.dump(self.memory, output_file)
            
    #Loads the Boltzmann temperature and total timesteps
    def load_config(self, config_fname):
        with open(config_fname, "rb") as input_file:
            # [self.T,self.epsilon, total_timesteps] = pickle.load(input_file)
            [self.T, total_timesteps] = pickle.load(input_file)
        return total_timesteps
      
    #Saves the last Boltzmann temperature and total timesteps
    def save_config(self, total_timesteps, config_fname):
        with open(config_fname, "wb") as output_file:
            # pickle.dump([self.T,self.epsilon, total_timesteps], output_file)
            pickle.dump([self.T, total_timesteps], output_file)

    #Loads the model, and the target weights for retraining
    def load(self, fname, tfname="", retrain=False):
        self.model = load_model(fname)
        if retrain:
            self.target_model.load_weights(tfname) 
            
    #Saves the model, and the target weights for retraining
    def save(self, fname, tfname="", backup = False):
        self.model.save(fname)
        if backup:
            self.target_model.save_weights(tfname)

