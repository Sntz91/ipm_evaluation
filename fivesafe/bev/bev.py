import numpy as np
import json
import cv2
from math import cos, sin

# TODO Test Pedestrians etc. also

SHIFT_DICT = {
    "person": [0, 0],
    "bicycle": [0.3, 0.3],
    "motorcycle": [0.3, 0.3],
    "car": [2.0, 0.8],
    "van": [1.1, 1.1],
    "truck": [5.0, 1.3],
    "bus": [5.0, 1.5]
}
        
class PositionEstimation:
    def __init__(self, H_fname: str, scale_factor: float) -> None:
        with open(H_fname, 'r') as file:
            self.H = np.array(json.load(file))
        self.scale_factor = scale_factor

    def transform(self, tracks, detections):
        """ Main Function Call for Fivesafe application. Calculate GCP for every track. """
        for track in tracks:
            mask = detections[track.detection_id-1].mask
            #world_position, psi_world, gcp_img = self.inverse_perspective_mapping(track, mask)
            world_position = self.calculate_ground_contact_point(track.label(), mask, track.xywh())
            track.xy = (world_position[0], world_position[1])
        return tracks
    
    def transform_w_rotated_bbox(self, tracks, rotated_bbox):
        pass

    def calculate_ground_contact_point(self, obj_class, mask, bb):
        """ Calculate Ground Contact Point """
        if obj_class in ('person'):
            return self._calculate_gcp_midpoint(bb)
        else:
            rotated_bbox = self._get_rotated_bbox(mask)
            bottom_edge_img, top_edge_img = self._find_bottom_top_edge(rotated_bbox)
            bottom_edge_world = self._transform_pts_image_to_world(bottom_edge_img)
            top_edge_world = self._transform_pts_image_to_world(top_edge_img)
            shift_vector_1 = self._get_norm_rotated_vector(bottom_edge_world, 90)
            shift_vector_2 = self._get_norm_rotated_vector(bottom_edge_world, -90)
            non_shifted_gcp = self._get_midpoint_from_edge(bottom_edge_world)
            shifted_gcp_c1 = self._shift_point_by_rvec_and_object_class(non_shifted_gcp, shift_vector_1, obj_class, rotated_bbox)
            shifted_gcp_c2 = self._shift_point_by_rvec_and_object_class(non_shifted_gcp, shift_vector_2, obj_class, rotated_bbox)
            shifted_gcp_c1_image = self._transform_pts_world_to_img(shifted_gcp_c1)
            shifted_gcp_c2_image = self._transform_pts_world_to_img(shifted_gcp_c2)
            if shifted_gcp_c1_image[0][1] < shifted_gcp_c2_image[0][1]:
                return shifted_gcp_c1[0]
            return shifted_gcp_c2[0]

    def _calculate_gcp_midpoint(self, bb):
        """ 
            Calculate GCP for Person. 
            It's the mid of the bottom edge of Bounding Box. 
        """
        gcp_img = np.array([[bb[0], bb[1] + bb[3]]])
        gcp_world = self._transform_pts_image_to_world(gcp_img)
        return gcp_world[0]
    
    def _find_bottom_top_edge(self, rotated_bbox: np.ndarray) -> tuple:
        """ 
        Find Bottom and Top Edge 
        Case 1: BBox is wider then high: Find further point to ref pt as bottom edge
        Case 2: BBox is higher then wide: Find closer point to ref pt as bottom edge
        """
        # Find Indices for Bottom- and Top Edge of Vehicle
        min_max_fn = np.argmin if self._is_bb_higher_than_wide(rotated_bbox) else np.argmax
        idx_bottom_vertex_1 = self._get_idx_bottom_vertex_1(
            rotated_bbox
        )
        idx_bottom_vertex_2 = self._get_idx_bottom_vertex_2(
            rotated_bbox, 
            idx_bottom_vertex_1,
            min_max_fn
        )
        idx_top_vertex_1, idx_top_vertex_2 = self._find_top_edge(idx_bottom_vertex_1, idx_bottom_vertex_2)
        # Get Values
        bottom_vertex_1 = rotated_bbox[idx_bottom_vertex_1]
        bottom_vertex_2 = rotated_bbox[idx_bottom_vertex_2]
        top_vertex_1 = rotated_bbox[idx_top_vertex_1]
        top_vertex_2 = rotated_bbox[idx_top_vertex_2]
        return np.array([bottom_vertex_1, bottom_vertex_2]), np.array([top_vertex_1, top_vertex_2])

    def _get_idx_bottom_vertex_1(self, rotated_bbox: np.ndarray) -> int:
        """ Get the Maximum y-value (the lowest point in image) """
        return np.argmax(rotated_bbox[:, 1])
    
    def _get_idx_bottom_vertex_2(self, rotated_bbox, idx_bottom_vertex_1, min_max_fn=np.argmax) -> int:
        """ Get the neighbouring point furthest away of vertex 1 """
        candidates = self._get_idx_neighbour_vertices(idx_bottom_vertex_1)
        lengths = self._get_neighbour_lengths(
            rotated_bbox, 
            idx_bottom_vertex_1, 
            candidates
        )
        return candidates[min_max_fn(lengths)]
    
    def _find_top_edge(self, idx_bottom_vertex_1, idx_bottom_vertex_2):
        """ Use the remaining points """
        idxs = {0, 1, 2, 3}
        return idxs - {idx_bottom_vertex_1, idx_bottom_vertex_2}

    def _get_idx_neighbour_vertices(self, ref_vertex_idx: int, bbox_len:int=4) -> list[int]:
        """ Get the neigbouring vertices of a point """
        candidates = [(ref_vertex_idx-1) % bbox_len, (ref_vertex_idx+1) % bbox_len]
        return candidates

    def _get_neighbour_lengths(self, 
        rotated_bbox: np.ndarray, 
        idx_ref_vertex: int, 
        idx_neighbours: list[int]
    ) -> list:
        """ Get the lenghts of ref point to neighbouring points """
        return [np.linalg.norm(rotated_bbox[idx_ref_vertex] - rotated_bbox[neighbour]) 
                for neighbour in idx_neighbours]

    def _is_bb_higher_than_wide(self, rotated_bbox: np.ndarray) -> bool:
        """ Check if Bounding Box is wider than high """
        # Like MARCEL, not sure why?
        idx_x_min, idx_x_max = np.argmin(rotated_bbox[:, 0]), np.argmax(rotated_bbox[:, 0])
        idx_y_min, idx_y_max = np.argmin(rotated_bbox[:, 1]), np.argmax(rotated_bbox[:, 1])
        width = rotated_bbox[idx_x_max][0] - rotated_bbox[idx_x_min][0]
        height = rotated_bbox[idx_y_max][1] - rotated_bbox[idx_y_min][1]
        return False if width > height else True

    def _shift_point_by_rvec_and_object_class(self, non_shifted_gcp, rvec, obj_class, rotated_bbox):
        lengths = SHIFT_DICT[obj_class]
        chosen_entry = 0 if self._is_bb_higher_than_wide(rotated_bbox) else 1
        length = lengths[chosen_entry]
        length = self.scale_factor * length
        shifted_gcp = non_shifted_gcp + rvec * length
        return np.expand_dims(shifted_gcp, axis=0)

    def _transform_pts_world_to_img(self, pts_world: np.ndarray) -> np.ndarray:
        """ Transform point from 3D world to 2D Image Coordinates (Z=0) """
        homogeneous_c = np.ones((pts_world.shape[0], 1))
        pts_world = np.hstack((pts_world, homogeneous_c))
        pts_world = np.linalg.inv(self.H) @ pts_world.T
        pts_world = pts_world / pts_world[2]
        return pts_world[:-1].T

    def _transform_pts_image_to_world(self, pts_img: np.ndarray) -> np.ndarray:
        """ Transform point from 2D image to 3D World Coordinates (Z=0) """
        homogeneous_c = np.ones((pts_img.shape[0], 1))
        pts_img = np.hstack((pts_img, homogeneous_c))
        pts_world = self.H @ pts_img.T
        pts_world = pts_world / pts_world[2]
        return pts_world[:-1].T

    def _get_norm_rotated_vector(self, bottom_edge, degree):
        """ Get Vector between two points and move it by certain degree """
        theta = np.deg2rad(degree)
        rotation_matrix = np.array([
            [cos(theta), sin(theta)],
            [-sin(theta), cos(theta)]
        ], dtype=np.float32)
        vector = bottom_edge[1] - bottom_edge[0] 
        vector_norm = vector / np.linalg.norm(vector)
        if vector_norm[0] < 0: # WHY?
            vector_norm = vector_norm * -1
        rotated_vector = rotation_matrix@vector_norm
        return rotated_vector

    def _get_midpoint_from_edge(self, edge: np.ndarray) -> np.ndarray:
        """ Return Midpoint of edge (two points) """
        midpoint_x = (edge[0][0] + edge[1][0]) / 2
        midpoint_y = (edge[0][1] + edge[1][1]) / 2
        return np.array([midpoint_x, midpoint_y])

    def _get_rotated_bbox(self, mask):
        """ Given a mask, return rotated BBOX """
        min_rect = cv2.minAreaRect(np.array(mask))
        angle_bb = min_rect[2]
        min_rect_pts = cv2.boxPoints(min_rect)
        min_rect_pts = np.int0(min_rect_pts)
        return min_rect_pts