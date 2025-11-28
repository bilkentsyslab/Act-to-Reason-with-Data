import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pylab as plb
plb.rcParams['font.size'] = 12

###########################
# Parameters to configure #
###########################
#path = os.path.split(os.getcwd())[0]+'/Act to Reason - Original/experiments/some_title/level2/training/training_data/' # Path of the training files of interest

model = "dynamic" # level1, level2, level3, dynamic
path = os.path.split(os.getcwd())[0]+f'/ATR_Original/experiments/some_title/{model}/training/training_data/' # Path of the training files of interest

ends = [] # Episode numbers at which retraining occurs, for instance [1500]
FINAL_EP = 10000 # Number of training episodes
complete_rew_subplot = [4,4] # "Reward Per Episode" subplot structure (Every subplot covers 1000 episodes)
focused_subplot = [3,4] # Focused average episode subplot structure (Every subplot covers 100 episodes)
Nw_rew_vs_step = 1000 # Averaging window size for "Average Reward vs Step" plot
Nw_rew_vs_ep = 1 # Averaging window size for "Average Reward vs Episode" plot
Nw_rew_per_ep = 200 # Averaging window size for "Average Reward per Episode" plot
Nw_Qloss = 10000 # Averaging window size for "Q-Loss Progression" plot
###########################

dats = [["reward","collision","train_hist"]]

if len(ends) >= 1:
    for end in ends:
        temp = ["retrain_"+str(end//100)+"_reward",
                "retrain_"+str(end//100)+"_collision",
                "retrain_"+str(end//100)+"_train_hist"]
        dats.append(temp)

reward = []
collision = []
train_hist = []
for idx, dat in enumerate(dats):   #this runs ids as indexes from zero to len-1, dat is the data namely [["reward","collision","train_hist"] + whatever comes from re-training part
    if len(ends) >= 1:
        if idx == len(dats)-1:
            end = FINAL_EP
        elif (len(dats)-1) > idx:
            end = ends[idx]
    else:
        end = FINAL_EP
    
    for i,dat_file in enumerate(dat):
        with open(path+dat_file+'.dat') as f:    
            data = np.asarray(list(csv.reader(f, delimiter="\t")),dtype='float64')    #creates an arraylist using csvreader to read a expecting data seperated by spaces
            data = data[data[:,0]<end]
            if i > 0:
                data[:,2:] = np.around(data[:,2:],decimals=3)
            if idx == 0:            
                if i == 0:
                    reward = data
                elif i == 1:
                    collision = data
                else:
                    train_hist = data
            else:
                if i == 0:
                    reward = np.concatenate((reward,data))
                elif i == 1:
                    collision = np.concatenate((collision,data))
                else:
                    train_hist = np.concatenate((train_hist,data))                    
                    
if len(ends) >= 1:            
    np.savetxt(path+'merged_reward.dat', reward, delimiter='\t')
    np.savetxt(path+'merged_collision.dat', collision, delimiter='\t')
    np.savetxt(path+'merged_train_hist.dat', train_hist, delimiter='\t')

# with open(path+"merged_reward.dat") as f:
#     reward = np.asarray(list(csv.reader(f, delimiter="\t")),dtype='float64')
# with open(path+"merged_collision.dat") as f:
#     collision = np.asarray(list(csv.reader(f, delimiter="\t")),dtype='float64')
# with open(path+"merged_train_hist.dat") as f:
#     train_hist = np.asarray(list(csv.reader(f, delimiter="\t")),dtype='float64')

reward_raw = reward.copy()
reward_df = pd.DataFrame(data=reward_raw,columns=["Episode","Step","Reward"])
reward_df_eps = reward_df[reward_df.duplicated(subset='Episode',keep='first')==False].Episode.values
reward2 = np.zeros((len(reward_df_eps),2))
for episode in reward_df_eps:
    episode = int(episode)
    reward2[episode,0] = episode 
    reward2[episode,1] = reward_df.loc[reward_df["Episode"]==episode].Reward.mean()

reward_ave = np.zeros((reward_df.shape[0] ,2)) # 2 columns, rows are the same with reward_df
reward = np.zeros((len(reward_df_eps),2))
prev_ep = 0
for episode,step,step_reward in zip(reward_df["Episode"],reward_df["Step"],reward_df["Reward"]):
    episode = int(episode)
    step = int(step)-1
    reward_ave[step,0] = step
    if step == 0:
        reward_ave[step,1] = step_reward
    else:
        reward_ave[step,1] = reward_ave[step-1,1] + (step_reward - reward_ave[step-1,1]) / (step+1.0)
    if episode != prev_ep:
        reward[episode-1,0] = episode-1
        reward[episode-1,1] = reward_ave[step-1,1]
    prev_ep = episode
reward[-1,0] = prev_ep
reward[-1,1] = reward_ave[-1,1]

all_data = [reward,collision,reward2,train_hist];

fig = plt.figure(figsize=(20.0,10.0))
if Nw_rew_vs_step > 1:
    smoothened = np.cumsum(np.asarray(reward_ave[:,1]), dtype=float)
    smoothened[Nw_rew_vs_step:] = smoothened[Nw_rew_vs_step:] - smoothened[:-Nw_rew_vs_step]
    smoothened[Nw_rew_vs_step-1:] = smoothened[Nw_rew_vs_step- 1:] / Nw_rew_vs_step
    smoothened[:Nw_rew_vs_step-1] = np.divide(smoothened[:Nw_rew_vs_step-1],np.arange(Nw_rew_vs_step-1)+1.0)   
else:
    smoothened = np.asarray(reward_ave[:,1], dtype=float)
# smoothened = np.convolve(np.asarray(reward_ave[:,1]),np.ones(10)/10,mode='same')
plt.plot(np.asarray(reward_ave[:,0],dtype='int'),smoothened)
plt.title("Training: Average Reward Progression")
plt.ylabel("Average Reward")
plt.xlabel("Step")
plt.savefig(path+'Training Average Reward vs Step')

fig = plt.figure(figsize=(20.0,10.0))
if Nw_Qloss > 1:
    smoothened = np.cumsum(np.asarray(all_data[3][:,2]), dtype=float)
    smoothened[Nw_Qloss:] = smoothened[Nw_Qloss:] - smoothened[:-Nw_Qloss]
    smoothened[Nw_Qloss-1:] = smoothened[Nw_Qloss- 1:] / Nw_Qloss
    smoothened[:Nw_Qloss-1] = np.divide(smoothened[:Nw_Qloss-1],np.arange(Nw_Qloss-1)+1.0)   
else:
    smoothened = np.asarray(all_data[3][:,2], dtype=float)
# smoothened = np.convolve(np.asarray(all_data[3][:,2]),np.ones(10)/10,mode='same')
plt.scatter(np.asarray(all_data[3][:,1],dtype='int'),smoothened)
plt.title("Training: Q-Loss Progression")
plt.ylabel("Q-Loss")
plt.xlabel("Step")
plt.savefig(path+'Training Q-Loss Progression - Scatter')

fig = plt.figure(figsize=(20.0,10.0))
if Nw_Qloss > 1:
    smoothened = np.cumsum(np.asarray(all_data[3][:,2]), dtype=float)
    smoothened[Nw_Qloss:] = smoothened[Nw_Qloss:] - smoothened[:-Nw_Qloss]
    smoothened[Nw_Qloss-1:] = smoothened[Nw_Qloss- 1:] / Nw_Qloss
    smoothened[:Nw_Qloss-1] = np.divide(smoothened[:Nw_Qloss-1],np.arange(Nw_Qloss-1)+1.0)   
else:
    smoothened = np.asarray(all_data[3][:,2], dtype=float)
# smoothened = np.convolve(np.asarray(all_data[3][:,2]),np.ones(10)/10,mode='same')
plt.plot(np.asarray(all_data[3][:,1],dtype='int'),smoothened)
plt.title("Training: Q-Loss Progression")
plt.ylabel("Q-Loss")
plt.xlabel("Step")
plt.savefig(path+'Training Q-Loss Progression')

fig = plt.figure(figsize=(20.0,10.0))
plt.plot(np.asarray(all_data[3][:,1],dtype='int'),all_data[3][:,3],label="Layer-1")
plt.plot(np.asarray(all_data[3][:,1],dtype='int'),all_data[3][:,5],label="Layer-2")
plt.plot(np.asarray(all_data[3][:,1],dtype='int'),all_data[3][:,7],label="Layer-3")
plt.plot(np.asarray(all_data[3][:,1],dtype='int'),all_data[3][:,9],label="Layer-4")
plt.title("Training: Average Weight Progression")
plt.ylabel("Average Weight")
plt.xlabel("Step")
plt.legend()
plt.savefig(path+'Training Average Weight Progression')

fig = plt.figure(figsize=(20.0,10.0))
plt.plot(np.asarray(all_data[3][:,1],dtype='int'),all_data[3][:,4],label="Layer-1")
plt.plot(np.asarray(all_data[3][:,1],dtype='int'),all_data[3][:,6],label="Layer-2")
plt.plot(np.asarray(all_data[3][:,1],dtype='int'),all_data[3][:,8],label="Layer-3")
plt.plot(np.asarray(all_data[3][:,1],dtype='int'),all_data[3][:,10],label="Layer-4")
plt.title("Training: Average Bias Progression")
plt.ylabel("Average Bias")
plt.xlabel("Step")
plt.legend()
plt.savefig(path+'Training Average Bias Progression')

start = 0
end = FINAL_EP

fig = plt.figure(figsize=(20.0,10.0))
plt.plot(np.asarray(all_data[0][:,0],dtype='int'),all_data[0][:,1])
plt.title("Training: Average Reward vs Episode")
plt.ylabel("Reward")
plt.xlabel("Episode")
plt.savefig(path+'Training Average Reward')

fig = plt.figure(figsize=(20.0,10.0))
fig.suptitle("Training: Average Reward vs Episode, (Titles indicate Collision Rate)")
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
ax = fig.add_subplot(complete_rew_subplot[0],complete_rew_subplot[1],1)
ax.plot(np.asarray(all_data[0][:,0],dtype='int'),all_data[0][:,1])
collision = np.asarray(all_data[1][start:end,1])
ax.set(title=(str(np.sum(collision)/(end/100)))+'%')

for i in range((end//1000)):
    ax = fig.add_subplot(complete_rew_subplot[0],complete_rew_subplot[1],i+2)
    # if i == 5:    
    #     ax.plot(np.asarray(all_data[0][i*1000:end,0],dtype='int'),all_data[0][i*1000:end,1])
    #     collision = np.asarray(all_data[1][i*1000:end,1])
    #     ax.set(title=(str(np.sum(collision)/5))+'%')

    # else:
    #     ax.plot(np.asarray(all_data[0][i*1000:(i+1)*1000,0],dtype='int'),all_data[0][i*1000:(i+1)*1000,1])
    #     collision = np.asarray(all_data[1][i*1000:(i+1)*1000,1])
    #     ax.set(title=(str(np.sum(collision)/10))+'%')
    
    ax.plot(np.asarray(all_data[0][i*1000:(i+1)*1000,0],dtype='int'),all_data[0][i*1000:(i+1)*1000,1])
    collision = np.asarray(all_data[1][i*1000:(i+1)*1000,1])
    ax.set(title=(str(np.sum(collision)/10))+'%')
    fig.savefig(path+'Complete Average Reward')

fig = plt.figure(figsize=(20.0,10.0))
if Nw_rew_per_ep > 1:
    smoothened = np.cumsum(np.asarray(all_data[2][:,1]), dtype=float)
    smoothened[Nw_rew_per_ep:] = smoothened[Nw_rew_per_ep:] - smoothened[:-Nw_rew_per_ep]
    smoothened[Nw_rew_per_ep-1:] = smoothened[Nw_rew_per_ep- 1:] / Nw_rew_per_ep
    smoothened[:Nw_rew_per_ep-1] = np.divide(smoothened[:Nw_rew_per_ep-1],np.arange(Nw_rew_per_ep-1)+1.0)   
else:
    smoothened = np.asarray(all_data[2][:,1], dtype=float)
# smoothened = np.convolve(np.asarray(all_data[2][:,1]),np.ones(5)/5,mode='same')
plt.plot(np.asarray(all_data[2][:,0],dtype='int'),smoothened)


plt.plot([4000, 4000],[-3 , 5], color = "blue")
plt.plot([6000, 6000],[-3 , 5], color = "blue")
plt.plot([8000, 8000],[-3 , 5], color = "blue")
plt.plot([2000, 2000],[-3 , 5], color = "blue")

collision = np.asarray(all_data[1][start:end,1])
plt.title("Training: Average Reward vs Episode")
plt.ylabel("Reward per Episode")
plt.xlabel("Episode")
plt.savefig(path+'Training Reward per Episode')

fig = plt.figure(figsize=(20.0,10.0))
fig.suptitle("Training: Reward of Each Episode, (Titles indicate Collision Rate)")
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
ax = fig.add_subplot(complete_rew_subplot[0],complete_rew_subplot[1],1)
if Nw_rew_per_ep > 1:
    smoothened = np.cumsum(np.asarray(all_data[2][:,1]), dtype=float)
    smoothened[Nw_rew_per_ep:] = smoothened[Nw_rew_per_ep:] - smoothened[:-Nw_rew_per_ep]
    smoothened[Nw_rew_per_ep-1:] = smoothened[Nw_rew_per_ep- 1:] / Nw_rew_per_ep
    smoothened[:Nw_rew_per_ep-1] = np.divide(smoothened[:Nw_rew_per_ep-1],np.arange(Nw_rew_per_ep-1)+1.0)   
else:
    smoothened = np.asarray(all_data[2][:,1], dtype=float)
# smoothened = np.convolve(np.asarray(all_data[2][:,1]),np.ones(5)/5,mode='same')
ax.plot(np.asarray(all_data[2][:,0],dtype='int'),smoothened)
collision = np.asarray(all_data[1][start:end,1])
ax.set(title=(str(np.sum(collision)/(end/100)))+'%')

for i in range((end//1000)):
    ax = fig.add_subplot(complete_rew_subplot[0],complete_rew_subplot[1],i+2)
    # if i == 5:
    #     smoothened = np.convolve(np.asarray(all_data[2][i*1000:end,1]),np.ones(5)/5,mode='same')
    #     ax.plot(np.asarray(all_data[2][i*1000:end,0],dtype='int'),smoothened)
    #     collision = np.asarray(all_data[1][i*1000:end,1])
    #     ax.set(title=(str(np.sum(collision)/5))+'%')
    # else:
    #     smoothened = np.convolve(np.asarray(all_data[2][i*1000:(i+1)*1000,1]),np.ones(5)/5,mode='same')
    #     ax.plot(np.asarray(all_data[2][i*1000:(i+1)*1000,0],dtype='int'),smoothened)
    #     collision = np.asarray(all_data[1][i*1000:(i+1)*1000,1])
    #     ax.set(title=(str(np.sum(collision)/10))+'%')
    
    if Nw_rew_per_ep > 1:
        smoothened = np.cumsum(np.asarray(all_data[2][i*1000:(i+1)*1000,1]), dtype=float)
        smoothened[Nw_rew_per_ep:] = smoothened[Nw_rew_per_ep:] - smoothened[:-Nw_rew_per_ep]
        smoothened[Nw_rew_per_ep-1:] = smoothened[Nw_rew_per_ep- 1:] / Nw_rew_per_ep
        smoothened[:Nw_rew_per_ep-1] = np.divide(smoothened[:Nw_rew_per_ep-1],np.arange(Nw_rew_per_ep-1)+1.0)   
    else:
        smoothened = np.asarray(all_data[2][i*1000:(i+1)*1000,1], dtype=float)
    # smoothened = np.convolve(np.asarray(all_data[2][i*1000:(i+1)*1000,1]),np.ones(5)/5,mode='same')
    #ax.plot(np.asarray(all_data[2][i*1000:(i+1)*1000,0],dtype='int'),smoothened)
    collision = np.asarray(all_data[1][i*1000:(i+1)*1000,1])
    #ax.set(title=(str(np.sum(collision)/10))+'%')
    #fig.savefig(path+'Complete Reward per Episode')

for i in range((end//1000)):
    fig = plt.figure(figsize=(20.0,10.0))
    fig.suptitle("Training: Average Reward vs Episode, (Titles indicate Collision Rate)")
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    # if i == 5:
    #     for j in range(5):
    #         ax = fig.add_subplot(3,2,j+1)
    #         collision = np.asarray(all_data[1][j*100+i*1000:(j+1)*100+i*1000,1])
    #         ax.plot(np.asarray(all_data[0][j*100+i*1000:(j+1)*100+i*1000,0],dtype='int'),all_data[0][j*100+i*1000:(j+1)*100+i*1000,1])
    #         ax.set(title=(str(np.sum(collision)))+'%')    
    #         fig.savefig(path+'Episode_'+str(i*1000)+'_'+str(end))
    # else:
    #     for j in range(10):
    #         ax = fig.add_subplot(3,4,j+1)
    #         collision = np.asarray(all_data[1][j*100+i*1000:(j+1)*100+i*1000,1])
    #         ax.plot(np.asarray(all_data[0][j*100+i*1000:(j+1)*100+i*1000,0],dtype='int'),all_data[0][j*100+i*1000:(j+1)*100+i*1000,1])
    #         ax.set(title=(str(np.sum(collision)))+'%')
    #         fig.savefig(path+'Episode_'+str(i*1000)+'_'+str((i+1)*1000))
    for j in range(10):
        ax = fig.add_subplot(focused_subplot[0],focused_subplot[1],j+1)
        collision = np.asarray(all_data[1][j*100+i*1000:(j+1)*100+i*1000,1])
        #ax.plot(np.asarray(all_data[0][j*100+i*1000:(j+1)*100+i*1000,0],dtype='int'),all_data[0][j*100+i*1000:(j+1)*100+i*1000,1])
        #ax.set(title=(str(np.sum(collision)))+'%')
        #fig.savefig(path+'Episode_'+str(i*1000)+'_'+str((i+1)*1000))

if len(ends) >= 1:
    for idx in range(len(dats)):
        if idx == 0:
            if os.path.exists(path+"crash_history.csv"):
                crash_history_temp = pd.read_csv(path+"crash_history.csv")
                crash_history_temp = crash_history_temp.loc[crash_history_temp['Episode']<ends[0]];
                crash_history = crash_history_temp
        else:
            if idx == len(ends):
                end = FINAL_EP
            else:
                end = ends[idx]
            if os.path.exists(path+"retrain_"+str(ends[idx-1]//100)+"_crash_history.csv"):
                crash_history_temp = pd.read_csv(path+"retrain_"+str(ends[idx-1]//100)+"_crash_history.csv")
                crash_history_temp = crash_history_temp.loc[crash_history_temp['Episode']<end];
                crash_history = pd.concat([crash_history,crash_history_temp],ignore_index=True)

    crash_history.to_csv(path+"merged_crash_history.csv", index=None, header=True)

        
    for idx in range(len(dats)):
        if idx == 0:
            if os.path.exists(path+"ego.csv"):
                ego_temp = pd.read_csv(path+"ego.csv")
                ego_temp = ego_temp.loc[ego_temp['Episode']<ends[0]];
                ego = ego_temp
        else:
            if idx == len(ends):
                end = FINAL_EP
            else:
                end = ends[idx]
            if os.path.exists(path+"retrain_"+str(ends[idx-1]//100)+"_ego.csv"):
                ego_temp = pd.read_csv(path+"retrain_"+str(ends[idx-1]//100)+"_ego.csv")
                ego_temp = ego_temp.loc[ego_temp['Episode']<end];
                ego = pd.concat([ego,ego_temp],ignore_index=True)

    ego.to_csv(path+"merged_ego.csv", index=None, header=True)
