import numpy as np
import cv2 
from ..measurements import Measurement

# TODO NOT A MEASUREMENT SOMEHOW. GET RID OF MEASUREMENT CLASS TBH.

class Track(Measurement):
    def __init__(
        self, 
        xy: tuple, 
        rvec_x: float, 
        rvec_y: float,
        id: int, 
        detection_id: int, 
        threshold: int = 10
    ) -> None:
        self.id = id
        self.xy = xy
        self.rvec_x = rvec_x
        self.rvec_y = rvec_y
        self.detection_id = detection_id
        self.threshold = threshold
        self.vehicle_gt_id = [] 

    def __repr__(self) -> str:
        # TODO!
        return f'Track id: {self.id},  \
            detection_id: {self.detection_id}'
    
    def xywh(self):
        # TODO
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
        # TODO
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
