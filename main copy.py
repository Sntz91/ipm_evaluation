import matplotlib.pyplot as plt
import numpy as np
from load_data import get_data
from ultralytics import YOLO

ts = 5
image_pv, image_tv, image_seg, vehicles_pv, vehicles_tv = get_data(ts, base_url='/Users/tobias/ziegleto/data/5Safe/carla/circle/', w=1920, h=1080)

from fivesafe.object_detection import Detectorv8Seg, draw_detections
model = YOLO('yolov8n-seg.pt')
detector = Detectorv8Seg(model=model, classes_of_interest=['car', 'person', 'bicycle'])
detections = detector.detect(image_pv)
#for detection in detections:
image_pv = draw_detections(np.asarray(image_pv), detections, mask=True)


from fivesafe.image_tracking import Tracker
width, height, _ = image_pv.shape
tracker = Tracker(width, height)
tracks = tracker.track(detections)
print(tracks)


from fivesafe.bev import PositionEstimation
homography_fname = 'conf/homography_matrix.json'
pos_est = PositionEstimation(homography_fname, 81/5) 
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.imshow(image_pv)
ax2.imshow(image_tv)

for track in tracks:
    mask = detections[track.detection_id-1].mask
    world_position, _ = pos_est.map_entity_and_return_relevant_points(track, mask)
    world_position = (world_position[0], world_position[1])
    track.world_position = world_position
    print(track.world_position)
    ax2.scatter(track.world_position[0], track.world_position[1])


plt.show()