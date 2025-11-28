import numpy as np
import pandas as pd 

import pickle
import os 

# TODO: refactor print statements and save out of functions into the main

## Results without cutting off x > merge_end + 50
"""
road_end: 406.05 
on-ramp start: 26.29
merge start: 65.22
merge end: 213.66
highway distance mean: 17.85
highway distance std: 11.25
on-ramp distance mean: 19.68
on-ramp distance std: 12.59
init velocity mean: 10.02
init velocity std: 3.42
highway init velocity mean: 9.92
highway init velocity std: 3.35
on-ramp init velocity mean: 16.56
on-ramp init velocity std: 1.59
highway init distance mean: 29.34
highway init istance std: 18.49
on-ramp init distance mean: 27.93
on-ramp init distance std: 17.98
velocity mean: 6.82, velocity std: 3.31
highway vel mean: 6.90, highway std: 3.14
on-ramp vel mean: 6.35, on-ramp std: 4.19
velocity mean in [0.00, 0.10]: 1.54, std: 0.73
highway velocity mean in [0.00, 0.10]: 1.68, std: 0.79
on-ramp velocity mean in [0.00, 0.10]: 1.09, std: 0.51
velocity mean in [0.75, 1.00]: 11.17, std: 1.85
highway velocity mean in [0.75, 1.00]: 10.96, std: 1.62
on-ramp velocity mean in [0.75, 1.00]: 12.48, std: 2.54
velocity mean in [0.90, 1.00]: 13.01, std: 1.48
highway velocity mean in [0.90, 1.00]: 12.60, std: 1.24
on-ramp velocity mean in [0.90, 1.00]: 15.11, std: 1.49
"""

## Results with cutting off x > merge_end + 50
"""
road_end: 406.05
on-ramp start: 26.29
merge start: 65.22
merge end: 213.66
highway distance mean: 16.38
highway distance std: 10.62
on-ramp distance mean: 19.68
on-ramp distance std: 12.59
init velocity mean: 10.02
init velocity std: 3.42
highway init velocity mean: 9.92
highway init velocity std: 3.35
on-ramp init velocity mean: 16.56
on-ramp init velocity std: 1.59
highway init distance mean: 29.34
highway init istance std: 18.49
on-ramp init distance mean: 27.93
on-ramp init distance std: 17.98
velocity mean: 6.20, velocity std: 3.31
highway vel mean: 6.16, highway std: 3.08
on-ramp vel mean: 6.35, on-ramp std: 4.19
velocity mean in [0.00, 0.10]: 1.29, std: 0.65
highway velocity mean in [0.00, 0.10]: 1.36, std: 0.70
on-ramp velocity mean in [0.00, 0.10]: 1.09, std: 0.51
velocity mean in [0.75, 1.00]: 10.66, std: 2.20
highway velocity mean in [0.75, 1.00]: 10.24, std: 1.91
on-ramp velocity mean in [0.75, 1.00]: 12.48, std: 2.54
velocity mean in [0.90, 1.00]: 12.89, std: 1.74
highway velocity mean in [0.90, 1.00]: 12.18, std: 1.50
on-ramp velocity mean in [0.90, 1.00]: 15.11, std: 1.49
"""

filename = "src/RECONSTRUCTED trajectories-400-0415_NO MOTORCYCLES.csv"
cwd = os.getcwd()
path = os.path.join(cwd, filename)

csv = pd.read_csv(path) 

data = {}
data["frame"] = np.array(csv["Frame_ID"]).flatten()
data["id"] = np.array(csv["Vehicle_ID"]).flatten()
data["lane"] = np.array(csv["Lane_ID"]).flatten()
data["x"] = np.array(csv["Local_Y"]).flatten()
data["v"] = np.array(csv["Mean_Speed"]).flatten()
data["acc"] = np.array(csv["Mean_Accel"]).flatten()

def delete_at(at):
    data["frame"] = np.delete(data["frame"], at)
    data["id"] = np.delete(data["id"], at)
    data["lane"] = np.delete(data["lane"], at)
    data["x"] = np.delete(data["x"], at)
    data["v"] = np.delete(data["v"], at)
    data["acc"] = np.delete(data["acc"], at)

## there are some vehicles that go to non existent lane 999, deleting those frames
at = np.argwhere(data["lane"]==999)
delete_at(at)

## delete lanes 1 to 5
at = np.argwhere(np.logical_and(data["lane"]!=6, data["lane"]!=7)).flatten()
delete_at(at)

## also make lanes 6-7 into 0-1 for later use
## 0: road, 1: on-ramp #TODO dikkat
data["lane"] = data["lane"]-6

## values are in feets, converting to meters
data["x"] = data["x"] * 0.3048
data["v"] = data["v"] * 0.3048
data["acc"] = data["acc"] * 0.3048

## data starts at x=50.011510848m, so we substract that
min_x = np.min(data["x"])
data["x"] = data["x"] - min_x
# print("min x before subtraction: " + str(min_x))

## highway starts at 0.0m, ends at 406.05m
road_start = 0
road_end = np.max(data["x"])

## checking start of on-ramp
## on-ramp starts at 26.29m
at = np.argwhere(data["lane"]==1)
xs = data["x"][at]
ramp_start = np.min(xs)

## checking the merging region start and end
min_change_x = road_end
max_change_x = road_start
for curr_id in np.unique(data["id"]):
    at = np.argwhere(data["id"]==curr_id)

    l = data["lane"][at]
    x = data["x"][at]
    
    change = False
    change_at = -1

    for j in range(l.shape[0]-1):
        if l[j] == 1 and l[j+1] == 0:
            change = True
            change_at = j
            break
        
    if change == True:
        change_x = x[change_at]
        if change_x < min_change_x:
            min_change_x = change_x
        if change_x > max_change_x:
            max_change_x = change_x
        
merge_start = min_change_x[0]
merge_end = max_change_x[0]

## delete all data where x > merge_end + 50 as it is not of interest
road_len = merge_end + 50
at = np.argwhere(data["x"]>=road_len)
delete_at(at)

save = {"ramp_start": ramp_start,
        "merge_start": merge_start,
        "merge_end": merge_end,
        "road_len": road_len,
        "v_high": np.max(data["v"])}

print("max v: " + str(np.max(data["v"])))


## Calculates the mean/std distance between observable cars
def dist_stats():
    dists_0 = []
    dists_1 = []

    for frame in np.unique(data["frame"]):
        at = np.argwhere(data["frame"]==frame).flatten()
        
        ls_at = data["lane"][at]
        xs_at = data["x"][at]

        l_0  = np.argwhere(ls_at==0).flatten()
        l_1 = np.argwhere(ls_at==1).flatten()

        x_0 = np.sort(xs_at[l_0])
        x_1 = np.sort(xs_at[l_1])
        
        for i in range(len(x_0)-1):
            dists_0.append(x_0[i+1]-x_0[i])
        for i in range(len(x_1)-1):
            dists_1.append(x_1[i+1]-x_1[i])
        
    dists_0 = np.array(dists_0)
    dists_1 = np.array(dists_1)
    
    print("highway distance mean: %.2f" % np.mean(dists_0))
    save["road_dist_mean"] = np.mean(dists_0)
    print("highway distance std: %.2f" % np.std(dists_0))
    save["road_dist_std"] = np.std(dists_0)
    print("on-ramp distance mean: %.2f" % np.mean(dists_1))
    save["ramp_dist_mean"] = np.mean(dists_1)
    print("on-ramp distance std: %.2f" % np.std(dists_1))
    save["ramp_dist_std"] = np.std(dists_1)
    
    
## Calculates the mean/std distance of initial car spawns and the car in their front
## along with the mean/std velocity of initial car spawns
def stats_init():
    dists_0 = []
    dists_1 = []
    
    vels = []
    vels_0 = []
    vels_1 = []
    
    for curr_id in np.unique(data["id"]):
        at = np.argwhere(data["id"]==curr_id).flatten()
        frames = data["frame"][at]
        
        init_frame = np.min(frames)
        at = np.argwhere(data["frame"]==init_frame).flatten()
        ls_at = data["lane"][at]
        xs_at = data["x"][at]
        
        at = np.argwhere(np.logical_and(data["id"]==curr_id, data["frame"]==init_frame)).flatten()
        curr_x = data["x"][at]
        
        # if initial x of a car is not the minimum of its lane, it entered from 
        # a left lane  instead of spawning at the beginning of the road
        if curr_x != np.min(xs_at):
            continue
        
        curr_v = data["v"][at]
        curr_l = data["lane"][at]
        vels.append(curr_v)
        if curr_l == 0:
            vels_0.append(curr_v)
        elif curr_l == 1:
            vels_1.append(curr_v)

        l_0  = np.argwhere(ls_at==0).flatten()
        l_1 = np.argwhere(ls_at==1).flatten()

        x_0 = np.sort(xs_at[l_0])
        x_1 = np.sort(xs_at[l_1])
        
        if len(x_0) > 1:
            dists_0.append(x_0[1]-x_0[0])
        if len(x_1) > 1:
            dists_1.append(x_1[1]-x_1[0])
        
    dists_0 = np.array(dists_0)
    dists_1 = np.array(dists_1)
    
    print("init velocity mean: %.2f" % np.mean(vels))
    print("init velocity std: %.2f" % np.std(vels))
    print("highway init velocity mean: %.2f" % np.mean(vels_0))
    save["init_road_v_mean"] = np.mean(vels_0)
    print("highway init velocity std: %.2f" % np.std(vels_0))
    save["init_road_v_std"] = np.std(vels_0)
    print("on-ramp init velocity mean: %.2f" % np.mean(vels_1))
    save["init_ramp_v_mean"] = np.mean(vels_1)
    print("on-ramp init velocity std: %.2f" % np.std(vels_1))
    save["init_ramp_v_std"] = np.std(vels_1)
    
    print("highway init distance mean: %.2f" % np.mean(dists_0))
    save["init_road_dist_mean"] = np.mean(dists_0)
    print("highway init istance std: %.2f" % np.std(dists_0))
    save["init_road_dist_std"] = np.std(dists_0)
    print("on-ramp init distance mean: %.2f" % np.mean(dists_1))
    save["init_ramp_dist_mean"] = np.mean(dists_1)
    print("on-ramp init distance std: %.2f" % np.std(dists_1))
    save["init_ramp_dist_std"] = np.std(dists_1)
        
    
## Calculates the mean/std velocity of cars
def v_stats():
    l0_at  = np.argwhere(data["lane"]==0).flatten()
    l1_at = np.argwhere(data["lane"]==1).flatten()

    v0 = data["v"][l0_at]
    v1 = data["v"][l1_at]
    
    print("velocity mean: %.2f, velocity std: %.2f" % (np.mean(data["v"]), np.std(data["v"])))
    print("highway vel mean: %.2f, highway std: %.2f" % (np.mean(v0), np.std(v0)))
    save["road_v_mean"], save["road_v_std"] = np.mean(v0), np.std(v0)
    print("on-ramp vel mean: %.2f, on-ramp std: %.2f" % (np.mean(v1), np.std(v1)))
    save["ramp_v_mean"], save["ramp_v_std"] = np.mean(v1), np.std(v1)
    
    
## Finding mean/std of velocities in the given range
def v_stats_in_range(r):
    n = data["v"].shape[0]
    vs = np.sort(data["v"])[int(n*r[0]):int(n*r[1])] 
    v_mean, v_std = np.mean(vs), np.std(vs)
    print("velocity mean in [%.2f, %.2f]: %.2f, std: %.2f" % (r[0], r[1], v_mean, v_std))
    
    l0_at  = np.argwhere(data["lane"]==0).flatten()
    l1_at = np.argwhere(data["lane"]==1).flatten()
    v0 = data["v"][l0_at]
    v1 = data["v"][l1_at]
    
    n = v0.shape[0]
    vs = np.sort(v0)[int(n*r[0]):int(n*r[1])] 
    v0_mean, v0_std = np.mean(vs), np.std(vs)
    print("highway velocity mean in [%.2f, %.2f]: %.2f, std: %.2f" % (r[0], r[1], v0_mean, v0_std))
    n = v1.shape[0]
    vs = np.sort(v1)[int(n*r[0]):int(n*r[1])] 
    v1_mean, v1_std = np.mean(vs), np.std(vs)
    print("on-ramp velocity mean in [%.2f, %.2f]: %.2f, std: %.2f" % (r[0], r[1], v1_mean, v1_std))

    return v_mean, v_std, v0_mean, v0_std, v1_mean, v1_std


## Finding mean/std of headway distances in the given range
def dist_stats_in_range(r):
    dists_0 = []
    dists_1 = []

    for frame in np.unique(data["frame"]):
        at = np.argwhere(data["frame"]==frame).flatten()
        
        ls_at = data["lane"][at]
        xs_at = data["x"][at]

        l_0  = np.argwhere(ls_at==0).flatten()
        l_1 = np.argwhere(ls_at==1).flatten()

        x_0 = np.sort(xs_at[l_0])
        x_1 = np.sort(xs_at[l_1])
        
        for i in range(len(x_0)-1):
            dists_0.append(x_0[i+1]-x_0[i])
        for i in range(len(x_1)-1):
            dists_1.append(x_1[i+1]-x_1[i])
        
    dists = np.array(dists_0 + dists_1)
    n = dists.shape[0]
    dists = np.sort(dists)[int(n*r[0]):int(n*r[1])] 
    dist_mean, dist_std = np.mean(dists), np.std(dists)
    
    return dist_mean, dist_std

def mean_dist_road():
    dist_tot = 0
    f_count = 0
    
    l6_tot = 0
    l7_tot = 0
    p6count = 0
    p7count = 0
    for frame in np.unique(data["frame"]):
        f_count += 1
        at = np.argwhere(data["frame"]==frame).flatten()
        pop = at.shape[0]
        dist = 490/pop
        dist_tot += dist
        
        ls_at = data["lane"][at]
        l_6  = np.argwhere(ls_at==0).flatten()
        l_7 = np.argwhere(ls_at==1).flatten()
        
        l6pop = l_6.shape[0]
        l7pop = l_7.shape[0]
        
        if l6pop != 0:
            l6_tot += road_end/l6pop
            p6count += 1
        if l7pop > 1:
            l7_tot += (merge_end - ramp_start)/l7pop
            p7count += 1

    print("mean_dist: " + str(dist_tot/f_count))
    print("mean_dist6: " + str(l6_tot/p6count))
    print("mean_dist7: " + str(l7_tot/p7count))

if __name__ == "__main__":
    print("road_end: %.2f" % road_end)
    print("on-ramp start: %.2f" % ramp_start)
    print("merge start: %.2f" % merge_start)
    print("merge end: %.2f" % merge_end)
    
    dist_stats()
    stats_init()
    v_stats()
    v_mean, _, _, _, _, _ = v_stats_in_range([0,.1]) # bottom 10% to find V_LOW 
    save["v_low"] = v_mean
    v_mean, _, _, _, _, _ = v_stats_in_range([.75,1]) # top 25% to find V_NOM
    save["v_nom"] = v_mean
    v_mean, v_std, v0_mean, v0_std, v1_mean, v1_std = v_stats_in_range([.9,1]) # top 10% to find V_HIGH
    save["v_nom_road"], save["v_nom_ramp"], save["v_std_road"], save["v_std_ramp"] = v0_mean, v1_mean, v0_std, v1_std
    dist_mean, _ = dist_stats_in_range([0,.1]) # bottom 10% to find D_CLOSE
    save["dist_close"] = dist_mean
    dist_mean, _ = dist_stats_in_range([.9,1]) # top 10% to find D_FAR
    save["dist_far"] = dist_mean
    
    mean_dist_road()
    
    file = open('data/analysis/analysis_results.pickle', 'wb')
    pickle.dump(save, file)
    file.close()
    
## code that was refactored out
"""
print(np.mean(accs))
print("**" + str(np.std(accs)))

#plt.hist(data["v"], num_bins)
#plt.show() 


def exp_dist(x):
    lamb = 0.75
    if x < 0:
        return 0
    else:
        return lamb * (math.e ** (-lamb * x)) * 20000

xs = np.arange(-5,5.01,0.01)
exps = []
for x in xs:
    exps.append(exp_dist(x))

plt.plot(xs, exps)
plt.hist(accs, num_bins)
plt.show() 


# delete data after the road's end in simulation
at = np.argwhere(data["x"]>road_len)
delete_at(at)

def get_rear_front_is(ego_x, xs):
    rear_x = road_start
    front_x = road_end
    rear_i = -1
    front_i = -1
    for i, x in enumerate(xs):
        if x > rear_x and x < ego_x:
            rear_x = x
            rear_i = i
        if x < front_x and x >= ego_x:
            front_x = x
            front_i = i
            
    return (rear_i, rear_x), (front_i, front_x)

def get_obs(ego_id, ego_frame, ego_l, ego_x, ego_v):
    # get obs of hypothetical vehicle at (lane_id, x_coord) at given frame
    # ignore vehicle with v_id for the observation
    
    # an observation is:
    # relative distance and velocity to 
    # front and back vehicles on left, right and current lanes
    # end merging region counts as vehicle with 0 velocity

    # get the environment at given frame, excluding v_id
    at = np.argwhere(np.logical_and(frames==ego_frame, ids != ego_id)).flatten()
    ls_at = ls[at]
    xs_at = xs[at]
    data["v"]_at = data["v"][at]

    lft_at = np.argwhere(ls_at==ego_l-1)
    cntr_at = np.argwhere(ls_at==ego_l)
    rght_at = np.argwhere(ls_at==ego_l+1)
    
    lft_xs, cntr_xs, rght_xs = xs_at[lft_at], xs_at[cntr_at], xs_at[rght_at]
    lft_data["v"], cntr_data["v"], rght_data["v"] = data["v"]_at[lft_at], data["v"]_at[cntr_at], data["v"]_at[rght_at]
    
    lft_rear, lft_front = get_rear_front_is(ego_x, lft_xs)
    lft_rear_i, lft_rear_x = lft_rear
    lft_front_i, lft_front_x = lft_front
    lft_rear_v = 0 if lft_rear_i == -1 else lft_data["v"][lft_rear_i]
    lft_front_v = 0 if lft_front_i == -1 else lft_data["v"][lft_front_i]
    
    cntr_rear, cntr_front = get_rear_front_is(ego_x, cntr_xs)
    cntr_rear_i, cntr_rear_x = cntr_rear
    cntr_front_i, cntr_front_x = cntr_front
    cntr_rear_v = 0 if cntr_rear_i == -1 else cntr_data["v"][cntr_rear_i]
    cntr_front_v = 0 if cntr_front_i == -1 else cntr_data["v"][cntr_front_i]
    
    rght_rear, rght_front = get_rear_front_is(ego_x, rght_xs)
    rght_rear_i, rght_rear_x = rght_rear
    rght_front_i, rght_front_x = rght_front
    rght_rear_v = 0 if rght_rear_i == -1 else rght_data["v"][rght_rear_i]
    rght_front_v = 0 if rght_front_i == -1 else rght_data["v"][rght_front_i]
    
    # code for checking the merge end as a car. should it be here?
    if ego_l == 6 and rght_front_i == -1 and ego_x < merge_end:
        rght_front_x = merge_end
        rght_front_v = 0
    if ego_l == 7 and cntr_front_i == -1 and ego_x < merge_end:
        cntr_front_x = merge_end
        cntr_front_v = 0
        
    obs = np.zeros(12)
    obs[0] = lft_rear_x
    obs[1] = lft_rear_v
    obs[2] = lft_front_x
    obs[3] = lft_front_v
    obs[4] = cntr_rear_x
    obs[5] = cntr_rear_v
    obs[6] = cntr_front_x
    obs[7] = cntr_front_v
    obs[8] = rght_rear_x
    obs[9] = rght_rear_v
    obs[10] = rght_front_x
    obs[11] = rght_front_v

    return obs



"""