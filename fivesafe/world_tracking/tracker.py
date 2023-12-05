from .world_tracker import WorldSort
from .tracks import Tracks
import numpy as np

class Tracker:
    def __init__(self):
        self.world_tracker_vehicles = WorldSort()
        self.world_tracker_vrus = WorldSort()

    def track(self, tracks): 
        """ returns world_tracks_vehicles and world_tracks_vrus """
        tracks_vrus = Tracks()
        tracks_vehicles = Tracks()

        dets_world_vehicles = np.empty((0, 4))
        dets_world_vrus = np.empty((0, 4))
        for track in tracks: 
            if track.label_id in [2, 3, 5, 7]: 
                dets_world_vehicles = np.append(
                        dets_world_vehicles, 
                        np.array([[track.xy_world[0], track.xy_world[1], track.label_id, track.detection_id]]),
                        axis=0
                )
            else:
                dets_world_vrus = np.append(
                        dets_world_vrus, 
                        np.array([[track.xy_world[0], track.xy_world[1], track.label_id, track.detection_id]]), 
                        axis=0
                )
        trjs_vehicles = self.world_tracker_vehicles.update(dets_world_vehicles)
        trjs_vrus = self.world_tracker_vrus.update(dets_world_vrus)

        tracks_vehicles = tracks_vehicles.numpy_to_tracks(trjs_vehicles)
        tracks_vrus = tracks_vrus.numpy_to_tracks(trjs_vrus)

        return tracks_vrus, tracks_vehicles
