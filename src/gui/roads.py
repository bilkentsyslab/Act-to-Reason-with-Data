from src.gui.common import DEF_LANE_WIDTH, DEF_LANE_LENGTH, BASE_OFFSET
from src.gui.common import fill_rectangle_scaled, draw_line_scaled

from src.gui.common import ROAD_LEN, MERGE_START, MERGE_END, RAMP_START, CAR_LEN

class Lane():
    def __init__(self, vehicles=None, valid_segments=None, start=0, end=DEF_LANE_LENGTH):
        self.start = start
        self.end = end
        if vehicles == None:
            self.vehicles = []
        if valid_segments == None:
            self.valid_segments = [[start, end]]
        else: 
            self.valid_segments = valid_segments
            
    def draw(self, canvas, offset=(0,0)):
        for segment in self.valid_segments:
            fill_rectangle_scaled(canvas, (segment[0]+1,0+1), (segment[1]-1,DEF_LANE_WIDTH-1), 
                                  color=(.8,.8,.8), offset=offset)
            draw_line_scaled(canvas, (segment[0],0), (segment[0],DEF_LANE_WIDTH), offset=offset)
            draw_line_scaled(canvas, (segment[1],0), (segment[1],DEF_LANE_WIDTH), offset=offset)
            

class LaneDivider():
    def __init__(self, invalid_segments=[]):
        self.invalid_segments = invalid_segments
        
    def is_valid(self, value):
        for invalid_segment in self.invalid_segments:
            if invalid_segment[1] >= value and invalid_segment[0] <= value:
                return False
        return True
    
    def draw(self, canvas, offset=(0,0)):
        for segment in self.invalid_segments:
            draw_line_scaled(canvas, (offset[0]+segment[0],offset[1]), (offset[0]+segment[1],offset[1]))
            
        
class Road():
    def __init__(self, lanes, lane_dividers):
        self.lanes = lanes
        self.lane_dividers = lane_dividers
                
    def draw(self, canvas, offset=BASE_OFFSET):
        n_lanes = len(self.lanes)
        offset = [offset[0], offset[1]] # copy to not change global offset
        
        # draw leftmost border of the road
        draw_line_scaled(canvas, (self.lanes[0].start,0), (self.lanes[0].end,0), offset=offset)
        
        for i, lane in enumerate(self.lanes):
            self.lanes[i].draw(canvas, offset)
            offset[1] += DEF_LANE_WIDTH
            if i < n_lanes-1:
                self.lane_dividers[i].draw(canvas, offset)
                
        # draw rightmost border of the road
        draw_line_scaled(canvas, (self.lanes[n_lanes-1].start,0), (self.lanes[n_lanes-1].end,0), offset=offset)
        
def generate_I80_merger():
    # TODO: check + CAR LEN  logic
    lanes = [Lane(end=ROAD_LEN), Lane(valid_segments=[[RAMP_START,MERGE_END]],end=ROAD_LEN)]
    # lanes = [
    #     Lane(valid_segments=[[RAMP_START, MERGE_END]], end=ROAD_LEN),  # Lane 0: Merging ramp lane
    #     Lane(end=ROAD_LEN)  # Lane 1: Main road lane
    # ]
    lane_dividers = [LaneDivider([[0,MERGE_START], [MERGE_END,ROAD_LEN]])]
    return Road(lanes, lane_dividers)