import matplotlib.pyplot as plt
import numpy as np
from load_data import get_data, bb_to_2d, plot_hull
from ultralytics import YOLO
import cv2

from fivesafe.object_detection import Detectorv8Seg, draw_detections, Detection_w_mask, Detections
from fivesafe.image_tracking import Tracker, draw_tracks
from fivesafe.bev import PositionEstimation, draw_world_position

width, height = 1920, 1080
tracker = Tracker(width, height)
model = YOLO('yolov8n-seg.pt')
detector = Detectorv8Seg(model=model, classes_of_interest=['car', 'person', 'bicycle'])

for ts in range(5, 100):
    print(ts)
    image_pv, image_tv, image_seg, vehicles_pv, vehicles_tv = get_data(ts, base_url='/Users/tobias/ziegleto/data/5Safe/carla/circle/', w=1920, h=1080)

    detections = detector.detect(image_pv)

    vehicle_pv = vehicles_pv[0]
    x_min, y_min, x_max, y_max = bb_to_2d(np.asarray(vehicle_pv['bb']))
    xyxy = (x_min, y_min, x_max, y_max)
    hull = vehicle_pv['hull']
    #plot_hull(hull)
    #plt.show()
    #hull = np.vstack((hull, hull[0, :]))
    hull = hull[:, [1, 0]]
    gt_detection = Detection_w_mask(xyxy=xyxy, label_id=2, score=1.0, mask=hull)
    gt_detections = Detections()
    gt_detections.append_measurement(gt_detection)
    
    #for detection in detections:
    image_pv_det = np.asarray(image_pv)
    image_pv = np.asarray(image_pv)
    image_tv = np.asarray(image_tv)
    image_pv_det = cv2.cvtColor(image_pv_det, cv2.COLOR_BGR2RGB)
    image_pv = cv2.cvtColor(image_pv, cv2.COLOR_BGR2RGB)
    image_tv = cv2.cvtColor(image_tv, cv2.COLOR_BGR2RGB)
    tracks = tracker.track(gt_detections)
    draw_tracks(image_pv, tracks)

    used_detections = gt_detections
    draw_detections(image_pv_det, used_detections, mask=True)


    homography_fname = 'conf/homography_matrix.json'
    pos_est = PositionEstimation(homography_fname, 1)  # scaling?! 81/5

    for track in tracks:
        # problem
        #if detection:
            mask = used_detections[track.detection_id-1].mask
            # NOTE That the BBOX does not matter for our approach at all!!!!! We Do Hull to min_area_rect.
            world_position, _, _ = pos_est.map_entity_and_return_relevant_points(track, mask)
            world_position = (world_position[0], world_position[1])
            track.world_position = world_position
            print(world_position)
            draw_world_position(image_tv, track.world_position, track.id, size=5)
            draw_world_position(image_tv, vehicles_tv[0]['gcp'], 0, size=5)
        #else: 
        #    force_update

    # TODO World Tracking einfuegen


    cv2.imshow('pv det', image_pv_det)
    cv2.imshow('pv track', image_pv)
    cv2.imshow('tv', image_tv)
    key = cv2.waitKey(0)
    if key == ord('q'):
        break

