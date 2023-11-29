
import numpy as np
from ..measurements import Measurements
from .track import Track


class Tracks(Measurements):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        out = ''
        for track in self:
            out += f'{track.id}: {track}\n'
        return out

    def numpy_to_tracks(self, track_list: np.ndarray):
        for track_candidate in track_list:
            print(track_candidate)
            x, y, track_id, detection_id, score, label_id = track_candidate
            track_candidate = Track(
                xy=(x, y), 
                label_id=int(label_id),
                score=score, 
                id=int(track_id), 
                detection_id=int(detection_id)
            )
            self.append_measurement(track_candidate)
        return self

    def append_measurement(self, measurement) -> None:
        self.append(measurement)

    def get_world_positions(self):
        positions = []
        for track in self:
            positions.append(track.xy)
        return positions
