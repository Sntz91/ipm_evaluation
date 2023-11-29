import numpy as np
import cv2 
from ..measurements import Measurement

# TODO NOT A MEASUREMENT SOMEHOW. GET RID OF MEASUREMENT CLASS TBH.

class Track(Measurement):
    def __init__(
        self, 
        xy: tuple, 
        label_id: int, 
        score: int, 
        detection_id: int, 
        id: int, 
        threshold: int = 10
    ) -> None:
        self.id = id
        self.xy = xy
        self.label_id = label_id
        self.score = score
        self.detection_id = detection_id
        self.threshold = threshold
        self.vehicle_gt_id = [] 

    def __repr__(self) -> str:
        return f'Track id: {self.id}, class: {self.label()}, \
            score: {self.score:.2f}, box: {self.xywh()}, \
            detection_id: {self.detection_id}'
    
    def xywh(self):
        pass
    
    @staticmethod
    def check_collision(bbox_coord, img_parameter, threshold):
        if(
            bbox_coord > (0 + threshold) \
            and bbox_coord < (img_parameter - threshold)
        ):
            return False
        return True
    
    def get_dict(self):
        output = super.get_dict()
        output["detection_id"] = self.detection_id
        return output
    
    def draw_detection_id(
        self, 
        frame: np.ndarray, 
        color=(255, 0, 0), 
        offset=(0, 0)
    ) -> np.ndarray:
        return cv2.putText(
            frame, 
            f'id: {self.id}, d_id: {self.detection_id}',
            (int(self.xy[0] + offset[0]), int(self.xy[1]+offset[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
            cv2.LINE_AA
        )
