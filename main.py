import matplotlib.pyplot as plt
import numpy as np
from load_data import get_data, bb_to_2d, plot_hull
from ultralytics import YOLO
import cv2
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from fivesafe.object_detection import Detectorv8Seg, draw_detections, Detection_w_mask, Detections
from fivesafe.image_tracking import Tracker, draw_tracks
from fivesafe.bev import PositionEstimation, draw_world_position

from utils.point_matching import match_points, calculate_euclidean_distance
from utils.iou import calculate_iou

width, height = 1920, 1080
tracker = Tracker(width, height)
model = YOLO('yolov8n-seg.pt')
detector = Detectorv8Seg(model=model, classes_of_interest=['car', 'person', 'bicycle'])


IMAGE_SHAPE = (1080, 1920)
IOU_THRESHOLD = 0.75

def draw_iou():
    pass

errors_per_ts_per_frame = []

class Correspondence:
    def __init__(self):
        pass

    def add_prediction(self):
        pass

    def add_gt(self):
        pass

identity_switches = 0
track_vehicle_correspondences = {}

for ts in range(4, 100):
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

    #iou = calculate_iou(detections[0].mask, gt_detections[0].mask)
    # TODO EMPTY PREDICTIONS
    #print(iou)
    
    image_pv_det = np.asarray(image_pv)
    image_pv = np.asarray(image_pv)
    image_tv = np.asarray(image_tv)
    image_pv_det = cv2.cvtColor(image_pv_det, cv2.COLOR_BGR2RGB)
    image_pv = cv2.cvtColor(image_pv, cv2.COLOR_BGR2RGB)
    image_tv = cv2.cvtColor(image_tv, cv2.COLOR_BGR2RGB)
    tracks = tracker.track(gt_detections)
    draw_tracks(image_pv, tracks)



    homography_fname = 'conf/homography_matrix.json'
    pos_est = PositionEstimation(homography_fname, 4)  # scaling?! 81/5

    predicted_world_positions = []
    gt_world_positions = []
    gt_detections = Detections()

    for vehicle_pv, vehicle_tv in zip(vehicles_pv, vehicles_tv):
        x_min, y_min, x_max, y_max = bb_to_2d(np.asarray(vehicle_pv['bb']))
        xyxy = (x_min, y_min, x_max, y_max)
        hull = vehicle_pv['hull']
        hull = hull[:, [1, 0]]
        gt_detection = Detection_w_mask(xyxy=xyxy, label_id=2, score=1.0, mask=hull)
        gt_detection.vehicle_gt_id = vehicle_pv['id']
        gt_detection.world_position = np.asarray(vehicle_tv['gcp'])
        gt_detections.append_measurement(gt_detection)
    #print(gt_detections)
    used_detections = gt_detections
    draw_detections(image_pv_det, used_detections, mask=True)
    for track in tracks:
        mask = used_detections[track.detection_id-1].mask
        if track.id not in track_vehicle_correspondences:
            track_vehicle_correspondences[track.id] = []
        track_vehicle_correspondences[track.id].append(used_detections[track.detection_id-1].vehicle_gt_id)
        #if track.gt_vehicle_id:
        #    old_id = track.gt_vehicle_id

        #track.gt_vehicle_id.append(used_detections[track.detection_id-1].vehicle_gt_id)

        #if track.gt_vehicle_id != old_id:
        #    print('oha Identity switch')
            # count and be done with it?
            # TODO Make a class that tracks over timsteps... this should be the tracker object..

        
        # NOTE That the BBOX does not matter for our approach at all!!!!! We Do Hull to min_area_rect.
        print(mask)
        world_position, _, _ = pos_est.map_entity_and_return_relevant_points(track, mask)
        world_position = (world_position[0], world_position[1])
        track.world_position = world_position
        #print(f'track id: {track.id}, gt id: {track.gt_vehicle_id}') # Somehow combine track ids with gt ids
        #print(world_position)
        draw_world_position(image_tv, track.world_position, track.id, size=5)
        draw_world_position(image_tv, vehicles_tv[0]['gcp'], 0, size=5)
        predicted_world_positions.append([track.world_position[0], track.world_position[1]])

    for vehicle in vehicles_tv:
        gt_world_positions.append(vehicle['gcp'])

    predictions = np.asarray(predicted_world_positions)
    gt = np.asarray(gt_world_positions)

    
    # TODO Threshold? 
    matched_pairs, unmatched_predictions, unmatched_gt = match_points(predictions, gt)
    print('matched', matched_pairs)
    print('unmatched_preds (false positives)', unmatched_predictions) # false positives
    print('unmatched_gt (false negatives)', unmatched_gt) # false negatives
    errors_per_frame = []

    # save point correspondence!

    # On this point we have corresponding tracks with ids. Now how can we keep track of this?

    for pair in matched_pairs:
        # save track id and gt id correspondence.
        error = calculate_euclidean_distance(predictions, gt, scaling_factor=1) # TODO SCALING FACTOR
        errors_per_frame.append(error)
    print('errors', errors_per_frame)
    print('errors avg', np.mean(errors_per_frame))

    # WHAT ABOUT IF MORE GT? False Negatives. 
    
    
    errors_per_ts_per_frame.append(errors_per_frame)

    # TODO World Tracking einfuegen


    cv2.imshow('pv det', image_pv_det)
    cv2.imshow('pv track', image_pv)
    cv2.imshow('tv', image_tv)
    key = cv2.waitKey(0)
    if key == ord('q'):
        break


errors_per_ts_per_frame = np.asarray(errors_per_ts_per_frame)
errors_per_ts = errors_per_ts_per_frame.mean(axis=1)
print('errors_per_ts', errors_per_ts)
print('error overall', errors_per_ts.mean())
print(track_vehicle_correspondences) # here i can extract identity switches

# For Evaluation we Need:
### 1. False Positives (Unmatched Predictions) ✅ (per ts)
### 2. False Negatives (Unmatched Ground Truths) ✅ (per ts)
### 3. Identity Switches ❌ (needs to be per ts also, look at formula)
### 4. Distances ✅ (per ts)

### Q: Do I do hungarian for every single point or for whole trajectories?


# To calculate Identity Switches, we need the whole trajectories, over all timesteps for each object.