import cv2
import numpy as np
import yaml
from .bufferless_cap import VideoCapture
from functools import wraps
from time import time
from collections import namedtuple

class Dict2ObjParser:
    def __init__(self, nested_dict):
        self.nested_dict = nested_dict

    def parse(self):
        nested_dict = self.nested_dict
        if (obj_type := type(nested_dict)) is not dict:
            raise TypeError(f"Expected 'dict' but found '{obj_type}'")
        return self._transform_to_named_tuples("root", nested_dict)

    def _transform_to_named_tuples(self, tuple_name, possibly_nested_obj):
        if type(possibly_nested_obj) is dict:
            named_tuple_def = namedtuple(tuple_name, possibly_nested_obj.keys())
            transformed_value = named_tuple_def(
                *[
                    self._transform_to_named_tuples(key, value)
                    for key, value in possibly_nested_obj.items()
                ]
            )
        elif type(possibly_nested_obj) is list:
            transformed_value = [
                self._transform_to_named_tuples(f"{tuple_name}_{i}", possibly_nested_obj[i])
                for i in range(len(possibly_nested_obj))
            ]
        else:
            transformed_value = possibly_nested_obj

        return transformed_value

def run(cfg_name, start_fn, bufferless=True):
    with open(cfg_name, 'r') as file:
            cfg = yaml.safe_load(file)
    url = cfg['input']['url']
    if bufferless:
        cap = VideoCapture(url)
    else:
        cap = cv2.VideoCapture(url)
    cfg = Dict2ObjParser(cfg).parse()
    start_fn(cap, cfg) 

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

def calculate_euclidean_distance(pt1, pt2, scaling_factor=1):
    assert pt1.shape == pt2.shape, "points must be same dim."
    distance = np.sqrt(np.sum((pt1-pt2)**2))
    return distance * scaling_factor

def calculate_distance_matrix(arr1, arr2, scaling_factor=1):
    size_arr1 = arr1.shape[0]
    size_arr2 = arr2.shape[0]
    dist_matrix = np.zeros((size_arr1, size_arr2))
    for i in range(size_arr1):
        for j in range(size_arr2):
            dist_matrix[i, j] = calculate_euclidean_distance(arr1[i], arr2[j], scaling_factor)
    return dist_matrix 
