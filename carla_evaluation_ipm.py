from fivesafe.datasets.carla import VehicleDataset
from fivesafe.object_detection import draw_detection
from fivesafe.utilities.point_matching import calculate_euclidean_distance
from fivesafe.utilities import Dict2ObjParser
from fivesafe.bev import (
    PositionEstimation,
    draw_world_position,
    draw_rotated_bbox,
    draw_bottom_edge,
    draw_gcp
)
import numpy as np
import yaml
import cv2


def start(cfg):
    dataset = VehicleDataset(cfg.dataset.url)
    position_estimator = PositionEstimation(
        cfg.bev.homography,
        cfg.bev.scalefactor
    )
    errors = []

    cv2.namedWindow('pv', cv2.WINDOW_NORMAL)
    cv2.namedWindow('tv', cv2.WINDOW_NORMAL)

    # Main Loop
    for (image_pv, image_tv), (vehicles_pv, vehicles_tv) in dataset:
        detections_pv = dataset.get_gt_detections(vehicles_pv)
        # draw shifted edge?
        (gcp_predicted,
         rotated_bbox,
         gcp_img,
         bottom_edge_img,
         bottom_edge_world) = position_estimator.calculate_ground_contact_point(
            'car',
            detections_pv[0].mask,
            detections_pv[0].xywh(),
            debug=True
        )
        gt_world_position = np.asarray(dataset.get_gcps(vehicles_tv))
        gcp_gt = gt_world_position[0]
        error = calculate_euclidean_distance(
            gcp_gt,
            gcp_predicted,
            cfg.bev.scalefactor
        )
        print('ERROR: ', error)
        errors.append(error)
        # Drawing
        draw_detection(image_pv, detections_pv[0],
                       mask=True, draw_id=False, draw_score=False)
        draw_world_position(image_tv, gcp_gt, 0)
        draw_world_position(image_tv, gcp_predicted, 1)
        draw_rotated_bbox(image_pv, rotated_bbox)
        draw_gcp(image_pv, gcp_img)
        draw_bottom_edge(image_pv, bottom_edge_img)
        draw_bottom_edge(image_tv, bottom_edge_world)
        cv2.putText(
            image_tv,
            f'Error: {error*100:.2f}cm',
            (730, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            3,
            cv2.LINE_AA
        )
        cv2.imshow('pv', image_pv)
        cv2.imshow('tv', image_tv)

        if cv2.waitKey(1) == ord('q'):
            break

    errors = np.asarray(errors)
    print(errors)
    print('mean', f'{np.mean(errors)*100:.2f} cm')


if __name__ == '__main__':
    cfg_name = './conf/carla_dataset.yaml'
    with open(cfg_name, 'r') as file:
        cfg = yaml.safe_load(file)
        cfg = Dict2ObjParser(cfg).parse()
    start(cfg)
