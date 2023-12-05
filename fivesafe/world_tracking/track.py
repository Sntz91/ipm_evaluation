from ..measurements import Measurement

class Track(Measurement):
    def __init__(
        self, 
        xy: tuple, 
        label_id: int,
        rvec_x: float, 
        rvec_y: float,
        id: int, 
        detection_id: int
    ) -> None:
        self.id = id
        self.xy = xy
        self.label_id = int(label_id)
        self.rvec_x = rvec_x
        self.rvec_y = rvec_y
        self.detection_id = detection_id
        self.vehicle_gt_id = [] 
        self.width = 4.5
        self.height = 1.8

    def __repr__(self) -> str:
        # TODO!
        return f'Track id: {self.id},  \
            detection_id: {self.detection_id}, position: {self.xy}, label_id: {self.label_id}'
    
    def xywh(self):
        return self.xy[0], self.xy[1], self.width, self.height
    
    def get_dict(self):
        output = super.get_dict()
        output["detection_id"] = self.detection_id
        return output