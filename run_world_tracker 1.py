import cv2
from fivesafe.object_detection import find_detector_class
from fivesafe.image_tracking import Tracker, draw_track
from fivesafe.bev import PositionEstimation, draw_world_position, draw_vehicle_baseplate
from fivesafe.utilities import run 
from fivesafe.decision_module import decision_making as dm
from ultralytics import YOLO
from world_tracker import WorldSort
import numpy as np

def start(cap, homography_fname, top_view_fname, model_name):
    # Intialize Detector, Position Estimator, CV2 Windows, and read Top View Image
    model = YOLO(model_name) 
    DetectorClass = find_detector_class(model_name)
    detector = DetectorClass(model=model, classes_of_interest=['person', 'car', 'bicycle'])
    scalefactor = 17.814
    pos_est = PositionEstimation(homography_fname, scalefactor)     # TUM 632.26/13.5
    cv2.namedWindow("top_view", cv2.WINDOW_NORMAL)
    cv2.namedWindow("perspective_view", cv2.WINDOW_NORMAL)
    top_view_org = cv2.imread(top_view_fname) 


    # Read Frame, Initialize Image Tracker
    ret, frame = cap.read()
    height, width, _ = frame.shape
    tracker = Tracker(width, height)
    tv_height, tv_width, _ = top_view_org.shape


    #Initialize World Tracker
    worldtracker_vehicles = WorldSort()
    worldtracker_vrus = WorldSort()

    # Initialize Zones from Config
    CONTOURS_INT_PATH, COLOR_INT_PATH = dm.init_contours('conf/not_intended_paths_contours.yaml')
    CONTOURS_TURNING_RIGHT, COLOR_TURN_RIGHT = dm.init_contours('conf/turning_right_zone_contours.yaml')
    COLOR_STANDARD = (79, 79, 47)

    # Draw Zones into Top View 
    #dm.draw_polylines_in_top_view(top_view_org, CONTOURS_INT_PATH, color=COLOR_INT_PATH)
    #dm.draw_polylines_in_top_view(top_view_org, CONTOURS_TURNING_RIGHT, color=COLOR_TURN_RIGHT)

    filenamecount = 0
    # Loop over Timesteps
    while True:
        detections = detector.detect(frame)
        tracks = tracker.track(detections)
        top_view = top_view_org.copy()

        nr_of_objects = len(tracks)

        
        # TODO Insert World Tracking Module that takes an Image Track and calculates the World Track with the Perspective Mapping; return List of world tracks to draw in next loop
        dets_world_vehicles = np.empty((0, 3))
        dets_world_vrus = np.empty((0, 3))
        for track in tracks:
            mask = detections[track.detection_id-1].mask
            world_position, psi_world, gcp_img = pos_est.map_entity_and_return_relevant_points(track, mask)
            world_position = (world_position[0], world_position[1])
            track.xy_world = world_position
            # DEBUGGING: Draw Raw Detection before Filtering
            top_view = draw_world_position(top_view, (int(track.xy_world[0]), int(track.xy_world[1])), track.id, (255, 0, 0))
            if track.label_id ==2:
                dets_world_vehicles = np.append(dets_world_vehicles, np.array([[track.xy_world[0], track.xy_world[1], track.label_id]]), axis=0) 
            else:
                dets_world_vrus = np.append(dets_world_vrus, np.array([[track.xy_world[0], track.xy_world[1], track.label_id]]), axis=0) 
            if track.label_id:  # !=2 for cleaning cars
                try:
                    cv2.circle(frame, (int(gcp_img[0][0]), int(gcp_img[0][1])),5, (255, 255, 0), -1)
                    cv2.circle(frame, (int(gcp_img[1][0]), int(gcp_img[1][1])),5, (255, 255, 0), -1)
                except:
                    print("No GPCs in Image available")
                    continue
        trjs_vehicles = worldtracker_vehicles.update(dets_world_vehicles)
        trjs_vrus = worldtracker_vrus.update(dets_world_vrus)

        for trj in trjs_vrus:
            # Transform Midpoint of Track to World Coordinates
            world_position = (int(trj[0]), int(trj[1]))
            rvec_normed = (trj[3], trj[4])
            """
            # Color Code Situations
            color = COLOR_STANDARD
            if dm.is_pt_in_contours(world_position, CONTOURS_INT_PATH):
                color = COLOR_INT_PATH
            elif dm.is_pt_in_contours(world_position, CONTOURS_TURNING_RIGHT):
                color = COLOR_TURN_RIGHT
            if dm.is_crowded(nr_of_objects, 1):
                frame = dm.draw_crowded(frame, width, height)
                top_view = dm.draw_crowded(
                    top_view, 
                    tv_width, 
                    tv_height, 
                    thickness=40, 
                    font_thickness=5, 
                    font_scale=5, 
                    y_offset=200
                )
            """
            color = COLOR_STANDARD  #Delete if color coding is active
            # Draw in Perspective- and Top-View
            top_view = draw_world_position(top_view, world_position, trj[2], color)

        for trj in trjs_vehicles:
            # Transform Midpoint of Track to World Coordinates
            world_position = (int(trj[0]), int(trj[1]))
            rvec_normed = (trj[3], trj[4])
            """
            # Color Code Situations
            color = COLOR_STANDARD
            if dm.is_pt_in_contours(world_position, CONTOURS_INT_PATH):
                color = COLOR_INT_PATH
            elif dm.is_pt_in_contours(world_position, CONTOURS_TURNING_RIGHT):
                color = COLOR_TURN_RIGHT
            if dm.is_crowded(nr_of_objects, 1):
                frame = dm.draw_crowded(frame, width, height)
                top_view = dm.draw_crowded(
                    top_view, 
                    tv_width, 
                    tv_height, 
                    thickness=40, 
                    font_thickness=5, 
                    font_scale=5, 
                    y_offset=200
                )
            """
            color = COLOR_STANDARD  #Delete if color coding is active
            # Draw in Perspective- and Top-View
            top_view = draw_world_position(top_view, world_position, trj[2], color)
            try:
                top_view = draw_vehicle_baseplate(top_view, np.array([[trj[0]], [trj[1]]]), np.array([[rvec_normed[0]], [rvec_normed[1]]]), 4.5, 1.8, scalefactor, thickness=3)
   
            except:
                pass
        for track in tracks:
            frame = draw_track(frame, track, color=color, draw_detection_id=True)

        # Activate if you want to save the Visualization of the output
        #cv2.imwrite( "C:/Users/mum21730/Desktop/filter_imgs/frames/"+ "%06d.jpg" % filenamecount, frame)
        #cv2.imwrite( "C:/Users/mum21730/Desktop/filter_imgs/worldmap/"+ "%06d.jpg" % filenamecount, top_view)
        filenamecount += 1
        cv2.imshow("perspective_view", frame)
        cv2.imshow("top_view", top_view)
        cv2.waitKey(1)
        ret, frame = cap.read()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run('conf/video_LUMPI.yaml', start, bufferless=False)