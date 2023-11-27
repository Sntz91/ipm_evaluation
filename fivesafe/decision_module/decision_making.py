import yaml
import cv2
import numpy as np
from ..utilities import draw_contours

def get_contours(points):
    contours = []
    for points_ in points:
        contours.append(
            cv2.convexHull(
                np.array(points_).reshape((-1, 1, 2)), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
        )
    return contours

def draw_polylines_in_top_view(top_view, contours, thickness=3, color=(255, 0, 0)):
    top_view = draw_contours(top_view, contours, -1, color=color, thickness=thickness)

def is_pt_in_contours(point, contours):
    for contour in contours:
        if cv2.pointPolygonTest(contour, point, False) != -1:
            return True
    return False

def is_crowded(nr, threshold):
    if nr > threshold:
        return True
    return False

def draw_crowded(frame, width, height, thickness=15, font_thickness=2, font_scale=1, y_offset=50):
    frame = cv2.rectangle(
        frame, 
        (0, 0), 
        (width, height), 
        (0, 69, 255), 
        thickness
    )
    frame = cv2.putText(
        frame, 
        'CROWDED', 
        (int(0.5*width), y_offset),
        cv2.FONT_HERSHEY_SIMPLEX, 
        font_scale, 
        (0, 69, 255), 
        font_thickness,
        cv2.LINE_AA
    )
    return frame