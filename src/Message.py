# This implemets a container for message.
# © 2020 Cevahir Köprülü All Rights Reserved

class Message:
    def __init__(self, msg):
        self.msg = msg
        self.fs_d = msg[0] # Front Side Relative Distance
        self.fs_v = msg[1] # Front Side Relative Velocity
        self.fc_d = msg[2] # Front Center Relative Distance
        self.fc_v = msg[3] # Front Center Relative Velocity
        self.rs_d = msg[4] # Rear Side Relative Distance
        self.rs_v = msg[5] # Rear Side Relative Velocity
        self.velocity = msg[6] # Current Velocity
        self.lane = msg[7] #Current Lane
        self.dist_end_merging = msg[8] #Distance to end_merging_point        









