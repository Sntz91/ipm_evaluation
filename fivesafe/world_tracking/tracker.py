from .world_tracker import WorldSort
import numpy as np

class Tracker:
    def __init__(self):
        self.world_tracker_vehicles = WorldSort()
        self.world_tracker_vrus = WorldSort()

    def track(self, tracks): 
        """ returns world_tracks_vehicles and world_tracks_vrus """
        dets_world_vehicles = np.empty((0, 3))
        dets_world_vrus = np.empty((0, 3))
        for track in tracks: 
            if track.label_id == 2: # TODO or truck
                dets_world_vehicles = np.append(dets_world_vehicles, np.array([[track.xy_world[0], track.xy_world[1], track.label_id]]), axis=0) 
            else:
                dets_world_vrus = np.append(dets_world_vrus, np.array([[track.xy_world[0], track.xy_world[1], track.label_id]]), axis=0) 
        trjs_vehicles = self.world_tracker_vehicles.update(dets_world_vehicles)
        trjs_vrus = self.world_tracker_vrus.update(dets_world_vrus)

        return trjs_vrus, trjs_vehicles
