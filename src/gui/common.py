import cv2
import pickle
import numpy as np



## constants
dt = 0.1
half_dtsqr = 0.5*(dt**2)
CAR_LEN = 5
MIN_DV = -4.5
MAX_DV = 3

## constants obtained from data analyis
file = open('data/analysis/analysis_results.pickle', 'rb')
data = pickle.load(file)
file.close()

#print(data)

VEL_NOM_ROAD = data["v_nom_road"]
VEL_STD_ROAD = data["v_std_road"]
VEL_NOM_RAMP = data["v_nom_ramp"]
VEL_STD_RAMP = data["v_std_ramp"]
VEL_LOW = 2#data["v_low"]
#VEL_NOM = data["v_nom"]
VEL_HIGH = data["v_high"]
 
DIST_NOM_ROAD = 34.96 # data["road_dist_mean"] #
DIST_STD_ROAD = data["road_dist_std"] / 2
DIST_NOM_RAMP = 53.35 #data["ramp_dist_mean"] #
DIST_STD_RAMP = data["ramp_dist_std"] / 2
DIST_CLOSE = 3 # data["dist_close"]
DIST_FAR = 30 # data["dist_far"]

ROAD_LEN = data["road_len"] 
RAMP_START = data["ramp_start"] 
MERGE_START = data["merge_start"] 
MERGE_END = data["merge_end"] 

VEL_INIT_NOM_ROAD = VEL_NOM_ROAD
VEL_INIT_STD_ROAD = VEL_STD_ROAD
VEL_INIT_NOM_RAMP = VEL_NOM_RAMP
VEL_INIT_STD_RAMP = VEL_STD_RAMP
DIST_INIT_NOM_ROAD = DIST_NOM_ROAD
DIST_INIT_STD_ROAD = DIST_STD_ROAD
DIST_INIT_NOM_RAMP = DIST_NOM_RAMP
DIST_INIT_STD_RAMP = DIST_STD_RAMP

"""
VEL_INIT_NOM_ROAD = data["init_road_v_mean"]
VEL_INIT_STD_ROAD = data["init_road_v_std"]
VEL_INIT_NOM_RAMP = data["init_ramp_v_mean"]
VEL_INIT_STD_RAMP = data["init_ramp_v_std"]
DIST_INIT_NOM_ROAD = data["init_road_dist_mean"]
DIST_INIT_STD_ROAD = data["init_road_dist_std"]
DIST_INIT_NOM_RAMP = data["init_ramp_dist_mean"]
DIST_INIT_STD_RAMP = data["init_ramp_dist_std"]
"""


def print_stats():
    print(f"D_CLOSE: {DIST_CLOSE}")
    print(f"VEL_LOW: {VEL_LOW}")
    print(f"DIST_NOM_ROAD: {DIST_NOM_ROAD}")
    print(f"DIST_STD_ROAD: {DIST_STD_ROAD}")
    print(f"DIST_NOM_RAMP: {DIST_NOM_RAMP}")
    print(f"DIST_STD_RAMP: {DIST_STD_RAMP}")
    print(f"ROAD_LEN: {ROAD_LEN}")
    print(f"ROAD_LEN: {ROAD_LEN}")

#print_stats()

## commont functions
def onehot(i, n):
    res = np.zeros(n)
    res[i] = 1
    return res

## common drawing code
DEF_LANE_WIDTH = 10
DEF_LANE_LENGTH = 100
CANVAS_W = 1500
CANVAS_H = 800
DRAW_SCALE = 5
BASE_OFFSET = (10, 10)

DEF_COLOR = (0,0,200)
EGO_COLOR = (0,200,0)

def draw_line_scaled(canvas, c1, c2, offset=(0,0), width=1, scale=DRAW_SCALE, color=(0,0,0)):
    cv2.line(canvas, 
             (round((c1[0]+offset[0])*scale),round((c1[1]+offset[1])*scale)), 
             (round((c2[0]+offset[0])*scale),round((c2[1]+offset[1])*scale)), 
             color, 
             round(width*scale))
    
def fill_rectangle_scaled(canvas, c1, c2, offset=(0,0), scale=DRAW_SCALE, color=(0,0,0)):
    cv2.rectangle(canvas, 
                  (round((c1[0]+offset[0])*scale),round((c1[1]+offset[1])*scale)), 
                  (round((c2[0]+offset[0])*scale),round((c2[1]+offset[1])*scale)), 
                  color, 
                  -1)

def draw_rectangle_scaled(canvas, c1, c2, offset=(0,0), width=1, scale=DRAW_SCALE, color=(0,0,0)):
    cv2.rectangle(canvas, 
                  (round((c1[0]+offset[0])*scale),round((c1[1]+offset[1])*scale)), 
                  (round((c2[0]+offset[0])*scale),round((c2[1]+offset[1])*scale)), 
                  color, 
                  round(width*scale))
    
def write_centered(canvas, c, text="", fontScale=.4, color=(0,0,0), thickness=1, offset=(0,0), scale=DRAW_SCALE):
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (round((c[0]+offset[0])*scale), round((c[1]+offset[0])*scale))
    cv2.putText(canvas, text, org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
    
# Define a color for the merge start marker
