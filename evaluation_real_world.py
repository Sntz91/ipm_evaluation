from fivesafe.datasets.vup1.dataset import Dataset
from fivesafe.object_detection import find_detector_class
from fivesafe.image_tracking import Tracker as ImageTracker, draw_tracks
from fivesafe.world_tracking import Tracker as WorldTracker
from fivesafe.utilities import Dict2ObjParser
from fivesafe.bev import (
    PositionEstimation,
    draw_world_positions,
    draw_world_position
)
from ultralytics import YOLO
import yaml
import cv2


def count_identity_switches(track_vehicle_correspondences):
    n_identity_switches = 0
    identity_switches = {}
    for track, correspondences in track_vehicle_correspondences.items():
        last_corr = correspondences[0]
        n_identity_switches_track = 0
        for correspondence in correspondences:
            if last_corr != correspondence:
                n_identity_switches += 1
                n_identity_switches_track += 1
            last_corr = correspondence
        identity_switches[track] = n_identity_switches_track
    return n_identity_switches, identity_switches


def start(cfg):
    dataset = Dataset(cfg.dataset.top_view_cfg,
                      cfg.dataset.perspective_view_cfg)
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
    position_estimator = PositionEstimation(
        cfg.bev.homography, cfg.bev.scalefactor)
    cv2.namedWindow("tv", cv2.WINDOW_NORMAL)
    cv2.namedWindow("pv1", cv2.WINDOW_NORMAL)
    cv2.namedWindow("pv2", cv2.WINDOW_NORMAL)

    # The loop is slow because we are not loading imgs into ram..
    for (image_pv, image_tv), (vehicles_pv, vehicles_tv) in dataset:
        img_camera1 = image_pv[0]  # only camera 2 at first
        vehicles_pv = vehicles_pv[0]
        print(vehicles_tv)
        detections = detector.detect(img_camera1)
        image_tracks = image_tracker.track(detections)
        image_tracks_transformed = position_estimator.transform(
            image_tracks, detections)
        world_tracks_vrus, world_Tracks_vehicles = world_tracker.track(
            image_tracks_transformed)
        draw_world_positions(image_tv, world_tracks_vrus, cfg.colors)
        draw_tracks(img_camera1, image_tracks,
                    cfg.colors, draw_detection_id=True)
        # TODO ROTATE AND MIDPOINT
        x, y = vehicles_tv['Rsp-BH3fW5']['x'], vehicles_tv['Rsp-BH3fW5']['y']
        draw_world_position(image_tv, [x, y], 0)

        cv2.imshow('pv1', img_camera1)
        cv2.imshow('tv', image_tv)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    cfg_name = './conf/vup1_dataset.yaml'
    with open(cfg_name, 'r') as file:
        cfg = yaml.safe_load(file)
        cfg = Dict2ObjParser(cfg).parse()
    start(cfg)
