from ..utilities.colors import COLORS
import cv2

def draw_world_position(top_view, position, track_id, color=None, size=35):
    if not color: 
        color = COLORS[track_id] 
    top_view = cv2.circle(top_view, (int(position[0]), int(position[1])), size, color, -1)
    return top_view
