from kafka import KafkaConsumer
from fivesafe.object_detection import Detection, Detections 
from fivesafe.image_tracking import Tracker as ImageTracker
from fivesafe.bev import PositionEstimation, draw_world_positions, draw_vehicle_baseplates
import json

consumer = KafkaConsumer('detected_objects_pp1_1', bootstrap_servers='17.11.1.21:19095')
#osition_estimator = PositionEstimation(
    # '/home/ziegleto/ziegleto/data/5Safe/LUMPI_Measurement1/homography_matrix.json',
    # 17.814
# )
width, height = 480, 450 # TODO
for msg in consumer:
    detections = Detections()
    # print (msg.value)
    r = json.loads(msg.value.decode())
    for detection in r['detectedObjects']:
        bb = detection['boundingBox']
        x, y = bb['x'], bb['y']
        b_width, b_height = bb['width'], bb['height']
        xyxy = [x, y, x+b_width, y+b_height]
        score = detection['confidence']
        label = detection['detectedClass']
        label_id = 1 # TODO
        detection_candidate = Detection(
            xyxy = xyxy,
            score = score,
            label_id = label_id
        )
        # Is from Interest? TODO
        detections.append_measurement(detection_candidate)
    print(detections)
    # print(r['detectedObjects'])
    break
consumer.close()
    # image_tracker = ImageTracker(width, height)
    # detections = []
    # image_tracks = image_tracker.track(detections)
    # Then do image tracking
    # Then do bev
    # Then do world tracking
    # profit.