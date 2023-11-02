import matplotlib.pyplot as plt
import numpy as np
from load_data import get_data, bb_to_2d, plot_hull, plot_3d_box
from ultralytics import YOLO
import cv2


def draw_psi(img, gcp, psi):
    cv2.line(img, gcp, psi, (0,0,255, 255), 2)

def draw_2d_bb(img, bb):
    x_min, y_min, x_max, y_max = bb_to_2d(np.asarray(bb))
    cv2.line(img, (int(x_min),int(y_min)), (int(x_max),int(y_min)), (0,0,255, 255), 2)
    cv2.line(img, (int(x_min),int(y_max)), (int(x_max),int(y_max)), (0,0,255, 255), 2)
    cv2.line(img, (int(x_min),int(y_min)), (int(x_min),int(y_max)), (0,0,255, 255), 2)
    cv2.line(img, (int(x_max),int(y_min)), (int(x_max),int(y_max)), (0,0,255, 255), 2)

def draw_3d_bb(img, bb):
    edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]
    for edge in edges:
        p1 = bb[edge[0]]
        p2 = bb[edge[1]]
        cv2.line(img, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (255,0,0, 255), 2) 

def draw_gcp(img, gcp):
    cv2.circle(img, gcp, 5, (0, 255, 0), -1)



cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE) 

for ts in range(5, 100):
    print(ts)
    image_pv, image_tv, image_seg, vehicles_pv, vehicles_tv = get_data(ts, base_url='/Users/tobias/ziegleto/data/5Safe/carla/circle/', w=1920, h=1080)


    vehicle = vehicles_pv[0]
    frame = np.asarray(image_pv)

    if ts>75:
        vehicle = vehicles_pv[0]
        frame = np.asarray(image_seg)
    elif ts>50:
        vehicle = vehicles_tv[0]
        frame = np.asarray(image_tv)
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    bb = vehicle['bb']
    gcp = vehicle['gcp']
    psi = vehicle['psi']

    draw_3d_bb(frame, bb)
    draw_2d_bb(frame, bb)
    draw_psi(frame, gcp, psi)
    draw_gcp(frame, gcp)
    # MIN AREA RECT TODO

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

