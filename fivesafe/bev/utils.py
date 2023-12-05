from ..utilities.colors import COLORS
import cv2
import numpy as np
from math import sin, cos

def draw_world_position(top_view, position, track_id, color=None):
    if not color: 
        color = COLORS[track_id] 
    top_view = cv2.circle(top_view, (int(position[0]), int(position[1])), 10, color, -1)
    return top_view

def draw_world_positions(top_view, tracks, colors, fixed_color=None):
    for track in tracks:
        color_ = getattr(colors, track.label())
        if fixed_color:
            color_ = fixed_color
        top_view = draw_world_position(top_view, track.xy, track.id, color=color_)
    return top_view

def draw_vehicle_baseplates(top_view, tracks, scalefactor, colors, thickness=10):
    for trj in tracks:
        color_ = getattr(colors, trj.label())
        world_position = (int(trj.xy[0]), int(trj.xy[1]))
        rvec_normed = (trj.rvec_x, trj.rvec_y)
        # Draw in Perspective- and Top-View
        top_view = draw_world_position(top_view, world_position, trj.id, color_)
        try: 
            top_view = draw_vehicle_baseplate(top_view, np.array([[trj.xy[0]], [trj.xy[1]]]), np.array([[rvec_normed[0]], [rvec_normed[1]]]), 4.5, 1.8, scalefactor, thickness=thickness)
        except:
            pass

    return top_view


def draw_vehicle_baseplate(top_view, midpoint, rvec, length, width, scalefactor, color=None, thickness=10):
    if not color: 
        color = (255, 255, 0)
    length_px = length * scalefactor
    width_px = width * scalefactor
    rvec_minus_dir = rvec * -1
    rvec_right = rotate_rvec_by_degree(rvec, 90)
    rvec_left = rvec_right * -1
    tl = midpoint + rvec * length_px/2 + rvec_left * width_px/2
    tr = tl + rvec_right * width_px
    bl = tl + rvec_minus_dir * length_px
    br = tr + rvec_minus_dir * length_px

    top_view = cv2.line(top_view, (int(tl[0]), int(tl[1])), (int(tr[0]), int(tr[1])), color, thickness)
    top_view = cv2.line(top_view, (int(bl[0]), int(bl[1])), (int(br[0]), int(br[1])), color, thickness)
    top_view = cv2.line(top_view, (int(tl[0]), int(tl[1])), (int(bl[0]), int(bl[1])), color, thickness)
    top_view = cv2.line(top_view, (int(tr[0]), int(tr[1])), (int(br[0]), int(br[1])), color, thickness)
    top_view = cv2.line(top_view, (int(tl[0]), int(tl[1])), (int(midpoint[0]), int(midpoint[1])), color, thickness)
    top_view = cv2.line(top_view, (int(tr[0]), int(tr[1])), (int(midpoint[0]), int(midpoint[1])), color, thickness)
    #top_view = cv2.circle(top_view, (int(tr[0]), int(tr[1])), 10, color, -1)
    #top_view = cv2.circle(top_view, (int(bl[0]), int(bl[1])), 10, color, -1)
    #top_view = cv2.circle(top_view, (int(br[0]), int(br[1])), 10, color, -1)

    return(top_view)

def rotate_rvec_by_degree(rvec, rotation_angle):
        
        #print("Rvec" + str(rvec_straight))
        rotation_angle = np.deg2rad(rotation_angle)
        rot_mat = np.array([[cos(rotation_angle), sin(rotation_angle)],
                           [-sin(rotation_angle), cos(rotation_angle)]], dtype=np.float32)
        
        rvec_straight_rotated = np.matmul(rot_mat, rvec)
       
        return rvec_straight_rotated
    
