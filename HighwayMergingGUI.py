"""
This file implements the visualization of simulated episodes
Set up the parameters in the main() function to visualize the desired episodes.
"""
# © 2021 Cevahir Köprülü All Rights Reserved

from tkinter import Tk, Canvas, Frame, BOTH, BOTTOM, TOP, StringVar, Label, PhotoImage
from tkinter.ttk import Button
import Params
import pandas as pd
import time
import pickle
import random
import numpy as np
from PIL import Image, ImageTk

class SimGUI(Frame):
    def __init__(self, *args):
        super(SimGUI,self).__init__()
        self.sim_df = args[0]
        self.episode_no = args[1]        
        self.root = args[2]
        self.show_ego = args[3]
        self.show_as_cars = args[4]
        
        self.proportion = 5.5
        self.fps = 10
        self.car_w = Params.Params.carwidth * self.proportion
        self.car_l = Params.Params.carlength * self.proportion
        self.lane_w = Params.Params.lanewidth * self.proportion
       
        self.start_x = 0
        self.start_y = 20
        self.GUI_start = Params.Params.start_environment
        self.GUI_end = Params.Params.init_size
        self.start_onramp = self.start_x + (Params.Params.start_onramp-self.GUI_start) * self.proportion
        self.end_road = self.start_x + (self.GUI_end-self.GUI_start) * self.proportion
        self.start_merging = self.start_x + (Params.Params.start_merging_point-self.GUI_start) * self.proportion
        self.end_merging = self.start_x + (Params.Params.end_merging_point-self.GUI_start) * self.proportion
        self.mid_y = self.start_y + self.lane_w
        self.end_y = self.start_y + self.lane_w * 2

        self.canvas_h = self.start_y + 2 * self.lane_w + self.start_y
        self.canvas_w = self.end_road + self.start_x
        
        if self.show_as_cars:
            self.car_images = {"orange": None,
                               "red": None,
                               "brown": None,
                               "blue": None,
                               "turk": None}
            for k in self.car_images:
                img_path = "C:/Users/SyslabUser/Desktop/car_"+str(k)+".png" 
                img = Image.open(img_path)
                img = img.resize((int(self.car_l), int(self.car_w)), Image.ANTIALIAS)
                self.car_images[k] = ImageTk.PhotoImage(img)
        
        self.step = 0
        
        self.cars = {}
        self.cars_status = {}

        self.levelk_colors = ["black","green","red","blue","purple"]
        self.show_ego_color = "purple"
        self.root.geometry(""+str(int(self.canvas_w)) + "x" + str(int(self.canvas_h)+100) + "+10+400")
        self.canvas = Canvas(self)

        self.initUI()

    def initUI(self):
        self.master.title("Episode: " + str(self.episode_no))
        self.pack(fill=BOTH, expand=1)

        # self.frame = Frame(height=start)
        self.canvas.create_rectangle(0,0,self.canvas_w,self.start_y,fill="green",outline="green")
        self.canvas.create_rectangle(0,self.start_y,self.canvas_w,self.end_y,fill="gray",outline="gray")

        self.canvas.create_line(self.start_onramp,self.mid_y,self.start_onramp,self.end_y,width=2) # first line


        self.canvas.create_line(self.start_x,self.start_y,self.end_road,self.start_y,width=2) # first line
        self.canvas.create_line(self.start_x,self.mid_y,self.start_merging,self.mid_y,width=2) # second line: start to start_merging
        self.canvas.create_line(self.start_x,self.end_y,self.end_merging,self.end_y,width=2) # third line

        self.canvas.create_line(self.start_merging, self.mid_y,self.end_merging+5*self.proportion, self.mid_y,dash=(4, 4),width=2) # second line: start_merging to end_merging 
        self.canvas.create_line(self.end_merging,self.end_y,self.end_merging+5*self.proportion,self.mid_y,width=2)
        self.canvas.create_line(self.end_merging+5*self.proportion,self.mid_y,self.end_road,self.mid_y,width=2)
        self.canvas.create_rectangle(0,self.end_y,self.canvas_w,self.canvas_h,fill="green",outline="green")
        self.canvas.create_polygon(self.end_merging+5*self.proportion,self.mid_y,self.canvas_w,self.mid_y,self.canvas_w,self.end_y,self.end_merging,self.end_y,fill="green",outline="green")

        self.button = Button(self,text="Start Sim",command=self.start_sim)
        self.step_text = StringVar()
        if not self.show_ego:
            self.step_text.set("Step " + str(self.step))
        else:
            self.step_text.set("Step " + str(self.step) + " || Car0 State: " + str(self.sim_df.iloc[0].State) + " || Car0 Velocity: " + "{:.2f}".format(self.sim_df.iloc[0].Velocity_X) + " km/h")
        self.step_label = Label(self, textvariable=self.step_text)

        self.step_label.pack(side=TOP)
        self.button.pack(side=BOTTOM)
        self.canvas.pack(fill=BOTH, expand=1)

    def start_sim(self):
        self.frame_no = 0
        self.master.title("Episode: " + str(self.episode_no))
        self.create_cars()
        self.update_sim()

    def create_cars(self):
        print("Create Cars - Step " + str(self.step))
        for dummy,car in self.sim_df.loc[(self.sim_df['Time_Step']==0)].iterrows():
            if car['Position_X'] >= self.GUI_start:
                if int(car['Car_ID']) == 0 and self.show_ego:
                    color =self.show_ego_color
                else:
                    color = self.levelk_colors[int(car["Level_k"])]
                delta_x = 0
                delta_y = 0
               
                if self.show_as_cars:
                    car_x = self.start_x + (car['Position_X']-self.GUI_start)*self.proportion - self.car_l/2
                    car_y = self.start_y + (1-car['Lane_ID'])*self.lane_w + self.lane_w/2
                    rnd_img_key = random.sample(self.car_images.keys(),1)[0]
                    car_rectangle = self.canvas.create_image(car_x, car_y, image=self.car_images[rnd_img_key])
                    if int(car['Level_k']) == 4:
                        car_rectangle_text = self.canvas.create_text(
                            car_x, car_y, font=("Arial bold", 14), text=str(int(car['Dynamic_Action'])))
                    else:
                        car_rectangle_text = self.canvas.create_text(
                            car_x, car_y, font=("Arial bold", 14), text=str(int(car['Level_k'])))
                else:
                    car_x = self.start_x + (car['Position_X']-self.GUI_start)*self.proportion - self.car_l
                    car_y = self.start_y + (1-car['Lane_ID'])*self.lane_w + (self.lane_w - self.car_w)/2
                    car_rectangle = self.canvas.create_rectangle(car_x, car_y, car_x + self.car_l, car_y + self.car_w, outline=color, fill=color)
    
                    if int(car['Level_k']) == 4:
                        car_rectangle_text = self.canvas.create_text(
                            car_x+self.car_l*0.5, car_y+self.car_w*0.5, 
                            font=("Arial bold", 14), text=str(int(car['Dynamic_Action'])))
                    else:
                        car_rectangle_text = self.canvas.create_text(
                            car_x+self.car_l*0.5, car_y+self.car_w*0.5, 
                            font=("Arial bold", 14), text=str(int(car['Level_k'])))
                
                car_rectangle_text_bg=self.canvas.create_rectangle(self.canvas.bbox(car_rectangle_text),fill="white")
                self.canvas.tag_lower(car_rectangle_text_bg,car_rectangle_text)
                if int(car['Car_ID']) == 0 and self.show_ego:
                    car_x = self.start_x + (car['Position_X']-self.GUI_start)*self.proportion - self.car_l
                    car_y = self.start_y + (1-car['Lane_ID'])*self.lane_w
                    ego_circle = self.canvas.create_oval(
                        car_x, car_y, car_x + self.car_l, 
                        car_y + self.lane_w, outline = "red", fill="", width = 2)
                    self.cars[int(car['Car_ID'])] = {"info": {"rect": car_rectangle,
                                                              "text": car_rectangle_text,
                                                              "text_bg": car_rectangle_text_bg,
                                                              "ego": ego_circle,
                                                              "dx": delta_x,
                                                              "dy": delta_y,
                                                              "pos_x": car['Position_X'],
                                                              "lane": car['Lane_ID']},
                                                     "status":True}
                else:
                    self.cars[int(car['Car_ID'])] = {"info": {"rect": car_rectangle,
                                                          "text": car_rectangle_text,
                                                          "text_bg": car_rectangle_text_bg,
                                                          "dx": delta_x,
                                                          "dy": delta_y,
                                                          "pos_x": car['Position_X'],
                                                          "lane": car['Lane_ID']},
                                                     "status":True}
                
        self.canvas.update()

    def add_car(self,car):
        # color = "orange"
        color = self.levelk_colors[int(car["Level_k"])]
        delta_x = 0
        delta_y = 0       
        if self.show_as_cars:
            car_x = self.start_x + (car['Position_X']-self.GUI_start)*self.proportion - self.car_l/2
            car_y = self.start_y + (1-car['Lane_ID'])*self.lane_w + self.lane_w/2
            rnd_img_key = random.sample(self.car_images.keys(),1)[0]
            car_rectangle = self.canvas.create_image(car_x, car_y, image=self.car_images[rnd_img_key])
            if int(car['Level_k']) == 4:
                car_rectangle_text = self.canvas.create_text(
                    car_x, car_y, 
                    font=("Arial bold", 14), text=str(int(car['Dynamic_Action'])))
            else:
                car_rectangle_text = self.canvas.create_text(
                    car_x, car_y, font=("Arial bold", 14), text=str(int(car['Level_k'])))
        else:
            car_x = self.start_x + (car['Position_X']-self.GUI_start)*self.proportion - self.car_l
            car_y = self.start_y + (1-car['Lane_ID'])*self.lane_w + (self.lane_w - self.car_w)/2

            car_rectangle = self.canvas.create_rectangle(
                car_x, car_y, car_x + self.car_l, car_y + self.car_w, outline=color, fill=color)
            if int(car['Level_k']) == 4:
                car_rectangle_text = self.canvas.create_text(
                    car_x+self.car_l*0.5, car_y+self.car_w*0.5, font=("Arial bold", 14), text=str(int(car['Dynamic_Action'])))
            else:
                car_rectangle_text = self.canvas.create_text(
                    car_x+self.car_l*0.5, car_y+self.car_w*0.5, font=("Arial bold", 14), text=str(int(car['Level_k'])))
        
        car_rectangle_text_bg=self.canvas.create_rectangle(self.canvas.bbox(car_rectangle_text),fill="white")
        self.canvas.tag_lower(car_rectangle_text_bg,car_rectangle_text)
        if int(car['Car_ID']) == 0 and self.show_ego:
            car_x = self.start_x + (car['Position_X']-self.GUI_start)*self.proportion - self.car_l
            car_y = self.start_y + (1-car['Lane_ID'])*self.lane_w
            ego_circle = self.canvas.create_oval(
                car_x, car_y, car_x + self.car_l, 
                car_y + self.lane_w, outline = "red", fill="", width = 2)
            self.cars[int(car['Car_ID'])] = {"info": {"rect": car_rectangle,
                                                      "text": car_rectangle_text,
                                                      "text_bg": car_rectangle_text_bg,
                                                      "ego": ego_circle,
                                                      "dx": delta_x,
                                                      "dy": delta_y,
                                                      "pos_x": car['Position_X'],
                                                      "lane": car['Lane_ID']},
                                             "status":True}
        else:
            self.cars[int(car['Car_ID'])] = {"info": {"rect": car_rectangle,
                                                  "text": car_rectangle_text,
                                                  "text_bg": car_rectangle_text_bg,
                                                  "dx": delta_x,
                                                  "dy": delta_y,
                                                  "pos_x": car['Position_X'],
                                                  "lane": car['Lane_ID']},
                                             "status":True}

    def update_sim(self):

        if not (self.step+1) > self.sim_df.iloc[-1].Time_Step:
            self.step += 1
            if not self.show_ego:
                self.step_text.set("Step " + str(self.step))
            else:
                temp_text = "Step " + str(self.step) + " || Car0 State: " + \
                    str(self.sim_df[self.sim_df['Time_Step']==self.step].iloc[0].State) + \
                        " || Car0 Velocity: " + "{:.2f}".format(
                            self.sim_df[self.sim_df['Time_Step']==self.step].iloc[0].Velocity_X * 3.6) + " km/h"
                self.step_text.set(temp_text)
            # i = 0
            for dummy,car in self.sim_df.loc[(self.sim_df['Time_Step']==self.step)].iterrows():
                car_id = int(car['Car_ID'])
                if car_id in self.cars.keys():
                    # if self.cars_status[i]==1 and self.cars[i][-2]>Params.Params.add_car_point:   
                    if self.cars[car_id]["status"] == True:
                        if self.cars[car_id]["info"]["pos_x"] > self.GUI_end:
                            self.canvas.delete(self.cars[car_id]["info"]["rect"])
                            self.canvas.delete(self.cars[car_id]["info"]["text"])
                            self.canvas.delete(self.cars[car_id]["info"]["text_bg"])
                            self.cars[car_id]["status"]  = False
                            self.cars[car_id]["info"] = {}
                        else:
                            self.cars[car_id]["info"]["dx"] = (car['Position_X']-self.cars[car_id]["info"]["pos_x"])*self.proportion/self.fps
                            self.cars[car_id]["info"]["dy"] = ((self.cars[car_id]["info"]["lane"] - car['Lane_ID'])*self.lane_w)/self.fps
                            self.cars[car_id]["info"]["pos_x"] = car['Position_X']
                            self.cars[car_id]["info"]["lane"] = car['Lane_ID']

                            if int(car["Level_k"]) == 4:
                                self.canvas.itemconfig(self.cars[car_id]["info"]["text"], text=str(int(car["Dynamic_Action"])))
                                
                                if not self.show_as_cars:
                                    if int(car['Car_ID']) == 0:
                                        if not self.show_ego:
                                            color = self.levelk_colors[int(car["Dynamic_Action"])]
                                            self.canvas.itemconfig(self.cars[car_id]["info"]["rect"], fill=color, outline=color)
                                    else:
                                        color = self.levelk_colors[int(car["Dynamic_Action"])]
                                        self.canvas.itemconfig(self.cars[car_id]["info"]["rect"], fill=color, outline=color)
                else: 
                    if car["Position_X"] > self.GUI_start:
                        self.add_car(car)

            self.move_frame()
        else:
            print("Step: " + str(self.step) + " - End")
            time.sleep(1)
            self.quit()
            self.destroy()

    def move_frame(self):
        if self.frame_no < self.fps:

            for car_id in self.cars.keys():
                if self.cars[car_id]["status"]==1:
                    dx = self.cars[car_id]["info"]["dx"]
                    dy = self.cars[car_id]["info"]["dy"]
                    self.canvas.move(self.cars[car_id]["info"]["rect"],dx,dy)
                    self.canvas.move(self.cars[car_id]["info"]["text"],dx,dy)
                    self.canvas.move(self.cars[car_id]["info"]["text_bg"],dx,dy)
                    if car_id == 0 and self.show_ego:
                        self.canvas.move(self.cars[car_id]["info"]["ego"],dx,dy)

            self.canvas.update()
            self.frame_no += 1
            self.after(int(250*Params.Params.timestep/self.fps), self.move_frame)
        else:
            self.frame_no = 0 
            self.update_sim()
def main():
    path = 'ATR_Original/experiments/some_title';
    directory = '/level1/simulation';
    
    num_pop_groups = 4 # Number of population groups simulated
    num_episodes = 200 # Number of episodes simulated for each population group(number of cars varies in that group)
    ego_type = "1_m99"
    vs_type = "0" # "1_m95"
    
    eps_file = "successful_eps.pickle"
    #eps_file = "long_eps.pickle"
    # eps_file = "crash_eps.pickle"
        
    #############################
    # Episode Selection Setting #
    #############################
    
    # Visualize an episode according to ego's final state    
    select_last_state = False
    desired_last_state = 2 # Check "car0_states" dictionary in Params.py
    
    # Visualize an episode according to ego's initial lane    
    select_ego_lane = False
    desired_ego_lane = 0
    
    # Visualize episodes with numbers between the following
    select_ep_num = False
    first_ep = 0 #14, 972, 976, 1026
    last_ep = 40
    
    # Visualize all pickled episodes
    select_all_eps = True
    
    # Visualize randomly selected episodes
    select_random_eps = True
    random_ep_num = 10
    #############################

    ################
    # Drawing Cars #
    ################
    show_ego = True # Show ego with a different color
    show_as_cars = False # Show cars with a given car picture (file path should be given in the class)
    ################
    
    
    csv_file = path + directory + "/simulation_level_"+ego_type+"_vs_L"+vs_type+"_"+\
        str(num_pop_groups)+"x"+str(num_episodes)+"eps_sim.csv"
    
    with open(path+directory+"/simulation_level_"+ego_type+"_vs_L"+vs_type+"_"+\
              str(num_pop_groups)+"x"+str(num_episodes)+"eps_"+eps_file,"rb") as handle:
        eps = pickle.load(handle)
    
    if select_ep_num:
        episode_nos = [ i for i in range(first_ep, last_ep+1)]
    elif select_all_eps:
        episode_nos = eps
    elif select_random_eps:
        episode_nos = random.sample(eps,random_ep_num)
    
    sim_df = pd.read_csv(csv_file)
    
    for episode_no in episode_nos:        
        
        episode_no = int(episode_no)
        ep_sim_df = sim_df[sim_df['Episode']==episode_no]
        
        if select_last_state:
            ego = ep_sim_df.loc[ep_sim_df['Car_ID']==0]
            print(ego.iloc[-1].State)
            
            if ego.iloc[-1].State == desired_last_state:
                print("Episode " + str(episode_no))
                demo(ep_sim_df,episode_no,show_ego,show_as_cars)
                
        elif select_ego_lane:
            ego = ep_sim_df.loc[ep_sim_df['Car_ID']==0]
            print(ego.iloc[0].Lane_ID)
            
            if ego.iloc[0].Lane_ID == desired_ego_lane:
                print("Episode " + str(episode_no))
                demo(ep_sim_df,episode_no,show_ego,show_as_cars)
                
        else:
            print("Episode " + str(episode_no))
            demo(ep_sim_df,episode_no,show_ego,show_as_cars)

def demo(ep_sim_df,episode_no,show_ego,show_as_cars):
    root = Tk()
    SimGUI(ep_sim_df,episode_no,root,show_ego,show_as_cars)
    root.mainloop()
    root.destroy()
    
if __name__ == '__main__':
    main()