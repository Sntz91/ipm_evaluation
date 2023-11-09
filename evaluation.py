from dataset import VehicleDataset, bb_to_2d
from fivesafe.image_tracking import Tracker
from fivesafe.object_detection import Detections, Detection_w_mask, draw_detections
from fivesafe.bev import PositionEstimation, draw_world_position
from ultralytics import YOLO
from utils.point_matching import match_points, calculate_euclidean_distance
import numpy as np
import cv2
from world_tracker import WorldSort

def get_gt_detections(vehicles):
    gt_detections = Detections()

    for vehicle in vehicles:
        hull = vehicle['hull']
        hull = hull[:, [1, 0]]
        detection = Detection_w_mask(
            xyxy = bb_to_2d(np.asarray(vehicle['bb'])),
            label_id = 2,
            score = 1.0,
            mask = hull
        )
        detection.vehicle_gt_id = vehicle['id']
        gt_detections.append_measurement(detection)

    return gt_detections


dataset = VehicleDataset(base_url='/Users/tobias/ziegleto/data/5Safe/carla/circle/')

IMAGE_SHAPE = (1080, 1920)
tracker = Tracker(IMAGE_SHAPE[1], IMAGE_SHAPE[0])
world_tracker = WorldSort()

track_vehicle_correspondences = {}
errors_per_ts_per_frame = []
identity_switches = 0

homography_fname = 'conf/homography_matrix.json'
pos_est = PositionEstimation(homography_fname, 5)  # scaling?! 81/5

def get_gcps(vehicles):
    gcps = []
    for vehicle in vehicles:
        gcps.append(vehicle['gcp'])
    return gcps


# For every timestep
for (image_pv, image_tv), (vehicles_pv, vehicles_tv) in dataset:
    gt_detections_pv = get_gt_detections(vehicles_pv)
    tracks = tracker.track(gt_detections_pv)

    # Why image tracking? better with IOU etc. When to open tracks etc.
    for track in tracks:
        # We determine the Min Area Rect with the mask
        # Surely, if there is no detection, we have no mask 
        corresponding_detection = gt_detections_pv[track.detection_id-1]
        if not corresponding_detection:
            # this does not necessarily mean that the track in image is lost btw.
            print('oops, no detection')
            continue
        else:
            mask = corresponding_detection.mask 

        world_position, _, _ = pos_est.map_entity_and_return_relevant_points(track, mask)
        track.world_position = (world_position[0], world_position[1])

        # Das muss ich dann spaeter eigentlich machen, nach world tracking TODO
        if track.id not in track_vehicle_correspondences:
            track_vehicle_correspondences[track.id] = []
        track_vehicle_correspondences[track.id].append(corresponding_detection.vehicle_gt_id) 
    

    gt_world_positions = np.asarray(get_gcps(vehicles_tv))
    predicted_world_positions = np.asarray(tracks.get_world_positions())
    
    # World Tracking insert here 
    trjs = world_tracker.update(predicted_world_positions)
    for trj in trjs:
        world_position = (int(trj[0]), int(trj[1]))
        draw_world_position(image_tv, world_position, 10, size=10)


    # Matching # TODO use the world track here
    matched_pairs, unmatched_predictions, unmatched_gt = match_points(predicted_world_positions, gt_world_positions)
    print('matched', matched_pairs)
    print('unmatched_preds (false positives)', unmatched_predictions) # false positives
    print('unmatched_gt (false negatives)', unmatched_gt) # false negatives


    # Calculate Error
    errors_per_frame = []
    pair_id = 0
    for pair in matched_pairs:
        error = calculate_euclidean_distance(pair[0], pair[1], scaling_factor=1)
        errors_per_frame.append(error)
        draw_world_position(image_tv, pair[0], pair_id, size=5)
        draw_world_position(image_tv, pair[1], pair_id+1, size=5)
        pair_id += 2
    errors_per_ts_per_frame.append(errors_per_frame)

    draw_detections(image_pv, gt_detections_pv, mask=True)

    cv2.imshow('pv', image_pv)
    cv2.imshow('tv', image_tv)
    key = cv2.waitKey(0)
    if key == ord('q'):
        break

errors_per_ts_per_frame = np.asarray(errors_per_ts_per_frame)
errors_per_ts = errors_per_ts_per_frame.mean(axis=1)
print('errors per ts', errors_per_ts)
print('error overall', errors_per_ts.mean())
print(track_vehicle_correspondences)