from fivesafe.datasets.carla import VehicleDataset
from fivesafe.object_detection import find_detector_class, draw_detections
from fivesafe.image_tracking import Tracker as ImageTracker, draw_tracks
from fivesafe.world_tracking import Tracker as WorldTracker
from fivesafe.utilities.point_matching import match_points, calculate_euclidean_distance 
from fivesafe.utilities import Dict2ObjParser
from fivesafe.bev import PositionEstimation, draw_world_positions
from ultralytics import YOLO
import numpy as np
import yaml
import cv2

# TODO Iou? dann brauchen wir aber auch perspective view maske. 

def start(cfg):
    dataset = VehicleDataset(cfg.dataset.url)
    h, w, _ = dataset.get_image_size()
    if not cfg.detection.use_gt:
        model = YOLO(cfg.detection.model)
        DetectorClass = find_detector_class(cfg.detection.model)
        detector = DetectorClass(
            model=model,
            classes_of_interest=cfg.detection.classes_of_interest
        )
    image_tracker = ImageTracker(w, h)
    world_tracker = WorldTracker()
    position_estimator  = PositionEstimation(cfg.bev.homography, cfg.bev.scalefactor)

    track_vehicle_correspondences = {}
    errors_per_ts_per_frame = []
    n_identity_switches = 0
    n_false_positives = 0
    n_false_negatives = 0 
    n_not_detected = 0

    for (image_pv, image_tv), (vehicles_pv, vehicles_tv) in dataset:
        if cfg.detection.use_gt:
            detections_pv = dataset.get_gt_detections(vehicles_pv)
        else:
            detections_pv = detector.detect(image_pv)
        image_tracks = image_tracker.track(detections_pv)
        image_tracks_transformed = position_estimator.transform(image_tracks, detections_pv)
        world_tracks_vrus, world_tracks_vehicles = world_tracker.track(image_tracks_transformed)
        
        # Matching TODO world_tracks vrus merged into vehicles again
        for track in world_tracks_vehicles:
            corresponding_detection = detections_pv[track.detection_id-1]
            # IF NO CORRESPONDING DETECTION, then what? Count, you will get them as false negatives anyways
            if not corresponding_detection:
                n_not_detected += 1
            if track.id not in track_vehicle_correspondences:
                track_vehicle_correspondences[track.id] = []
            track_vehicle_correspondences[track.id].append(corresponding_detection.vehicle_gt_id)

        gt_world_positions = np.asarray(dataset.get_gcps(vehicles_tv))
        predicted_world_positions = np.asarray(world_tracks_vehicles.get_world_positions())
        # Draw
        image_pv = draw_detections(image_pv, detections_pv, mask=True)
        draw_world_positions(image_tv, world_tracks_vehicles, cfg.colors) # TODO MAYBE DRAW ERROR LINE
        draw_tracks(image_pv, image_tracks, cfg.colors)
        # Evaluation scores
        matched_pairs, unmatched_predictions, unmatched_gt = match_points(predicted_world_positions, gt_world_positions)
        print('matched', matched_pairs)
        print('unmatched_preds (false positives)', unmatched_predictions) # false positives
        print('unmatched_gt (false negatives)', unmatched_gt) # false negatives
        n_false_positives += len(unmatched_predictions)
        n_false_negatives += len(unmatched_gt)

        errors_per_frame = []
        for pair in matched_pairs:
            error = calculate_euclidean_distance(pair[0], pair[1], scaling_factor=cfg.bev.scalefactor)
            errors_per_frame.append(error)
        errors_per_ts_per_frame.append(errors_per_frame)
        cv2.imshow('pv', image_pv)
        cv2.imshow('tv', image_tv)
        if cv2.waitKey(0) == ord('q'):
            break

    errors_per_ts_per_frame = np.asarray(errors_per_ts_per_frame)
    errors_per_ts = errors_per_ts_per_frame.mean(axis=1)
    print('errors per ts', errors_per_ts)
    print('error overall', errors_per_ts.mean())
    print('nr of identity switches TODO', n_identity_switches)
    print(track_vehicle_correspondences)
    print('nr of false positives', n_false_positives)
    print('nr of false negatives', n_false_negatives)
    print('nr of not deteced', n_not_detected)


if __name__=='__main__':
    cfg_name = './conf/carla_dataset.yaml'
    with open (cfg_name, 'r') as file:
        cfg = yaml.safe_load(file)
        cfg = Dict2ObjParser(cfg).parse()
    start(cfg)
