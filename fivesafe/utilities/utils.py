import cv2
import numpy as np
import yaml
from .bufferless_cap import VideoCapture
from functools import wraps
from time import time

def run(cfg_name, start_fn, bufferless=True):
    with open(cfg_name, 'r') as file:
            cfg = yaml.safe_load(file)
    url = cfg['capture_url']
    if bufferless:
        cap = VideoCapture(url)
    else:
        cap = cv2.VideoCapture(url)
    # Homography not necessary for detector, tracker.
    if 'homography' in cfg.keys():
        homography_fname = cfg['homography']
    if 'top_view' in cfg.keys():
        top_view_fname = cfg['top_view'] 
    model_name = cfg['model']

    start_fn(cap, homography_fname, top_view_fname, model_name)

def draw_contours(
    frame: np.array,
    contours: tuple,
    indices: int = -1, 
    thickness = 1, 
    color: tuple = (0, 0, 255),
    alpha: float = 0.3,
    ) -> np.ndarray:
    if alpha:
        mask = np.zeros(frame.shape, np.uint8)
        cv2.drawContours(mask, contours, indices, color, -1)
        frame[:] = cv2.addWeighted(mask, alpha, frame, beta=1.0, gamma=0.0)
    cv2.drawContours(frame, contours, indices, color, thickness)
    return frame

def draw_rectangle(
    frame: np.array,
    pt1: tuple,
    pt2: tuple,
    color=(255, 0, 0),
    thickness=1,
    alpha: float=0.6
    ) -> np.ndarray:
    if alpha:
        mask = np.zeros(frame.shape, np.uint8)
        cv2.rectangle(
               mask,
               (int(pt1[0]), int(pt1[1])),
               (int(pt2[0]), int(pt2[1])),
               color,
               -1
        )
        frame[:] = cv2.addWeighted(mask, alpha, frame, beta=1.0, gamma=0.0)
    cv2.rectangle(
        frame,
        (int(pt1[0]), int(pt1[1])),
        (int(pt2[0]), int(pt2[1])),
        color,
        thickness
    )
    return frame


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r took: %2.4f sec' % \
          (f.__name__, te-ts))
        return result
    return wrap

import socket

def connect_socket(url, port):
    # Connect Socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (url, port)
    print ('starting up on %s port %s' % server_address)
    sock.bind(server_address)
    sock.listen(1)
    connection, client_address = sock.accept()
    return connection, client_address