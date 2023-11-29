from torch.utils.data import Dataset
from ..object_detection import Detections, Detection_w_mask
import numpy as np
import pickle
import cv2
import torch
import os

class VehicleDataset(Dataset):
    def __init__(self, base_url, transform=None, target_transform=None):
        # read annotation file
        self.base_url = base_url
        self.transform = transform
        self.target_transform = target_transform

        pv_path = os.path.join(self.base_url, 'output/pv/data_new.pickle')
        tv_path = os.path.join(self.base_url, 'output/tv/data_new.pickle')

        with open(pv_path, 'rb') as handle:
            self.data_pv = pickle.load(handle)

        with open(tv_path, 'rb') as handle:
            self.data_tv = pickle.load(handle)

        assert len(self.data_pv) == len(self.data_tv), 'tv pv not same length?'

    def __len__(self):
        return len(self.data_pv)-2

    def __getitem__(self, idx):
        idx = idx+1 
        label_pv = self.data_pv[idx]
        label_tv = self.data_tv[idx]
        image_pv_url = self.base_url + label_pv[0]['img']
        image_tv_url = self.base_url + label_tv[0]['img']
        image_pv = cv2.imread(image_pv_url)
        image_tv = cv2.imread(image_tv_url)

        return (image_pv, image_tv), (label_pv, label_tv) 
    
    def get_image_size(self):
        image_pv_url = self.base_url + self.data_pv[1][0]['img']
        image_pv = cv2.imread(image_pv_url)
        return image_pv.shape
    
    @staticmethod
    def get_gt_detections(vehicles):
        gt_detections = Detections()

        for vehicle in vehicles:
            hull = vehicle['hull']
            hull = hull[:, [1, 0]]
            detection = Detection_w_mask(
                xyxy = bb_to_2d(np.asarray(vehicle['bb'])),
                label_id = 2,
                score = 1.0,
                mask = hull
            )
            detection.vehicle_gt_id = vehicle['id']
            gt_detections.append_measurement(detection)

        return gt_detections
    
    @staticmethod
    def get_gcps(vehicles):
        gcps = []
        for vehicle in vehicles:
            gcps.append(vehicle['gcp'])
        return gcps



class VehicleMaskPsiDataset(Dataset):
    """ 
        Dataset from carla; X: Vehicle Mask at t and t-1, y: psi vector 
        Note that it only works for one vehicle per image.
    """
    def __init__(self, base_url, n_mask_pts=65):
        self.base_url = base_url
        self.n_mask_pts = n_mask_pts
        pickle_path = os.path.join(self.base_url, 'data_new.pickle')
        with open(pickle_path, 'rb') as handle:
            self.vehicle_data = pickle.load(handle)

    def __len__(self):
        return len(self.vehicle_data)-1

    def __getitem__(self, idx: int) -> tuple: 
        """ 
            Return X, y given idx
            X: nr_of_masks * mask_len * point_dimension (b*65*2)
            y: psi vector (1*2)
        """
        label_t1 = self.vehicle_data[idx+1][0]
        label_t0 = self.vehicle_data[idx][0]
        # Mask
        mask_t1 = self._normalize_mask(label_t1['hull'])
        mask_t0 = self._normalize_mask(label_t0['hull'])
        psi = self._normalize_psi(label_t1['psi'], label_t1['gcp'])
        # nr_of_masks, mask_len, point_dimension
        masks = torch.stack((mask_t0, mask_t1))
        return masks, psi

    def _get_bb(self, idx: int) -> tuple:
        """ Return bbox (xyxy) coordinates """
        vehicle = self.vehicle_data[idx][0]
        return vehicle['bb']

    def _get_bb_img(self, idx: int) -> np.ndarray:
        """ Cut image from bbox """
        img = self._get_image(idx)
        bb = self._get_bb(idx) # here
        return img[bb[1]:bb[3], bb[0]:bb[2]]

    def _get_image(self, idx: int) -> np.ndarray:
        """ Return image, e.g., for visualization """
        label_pv = self.vehicle_data[idx+1]
        image_pv_url = self.base_url + label_pv[0]['img']
        image_pv = cv2.imread(image_pv_url)
        return image_pv
    
    def _normalize_psi(self, psi: list, gcp: list) -> torch.Tensor:
        """ Given psi and gcp, normalize psi to origin and length 1 """
        psi_normalized = [c - c0 for c, c0 in zip(gcp, psi)]
        psi_normalized = psi_normalized / np.sqrt(np.dot(psi_normalized, psi_normalized))
        return torch.Tensor(-psi_normalized)

    def _normalize_mask(self, mask: np.ndarray) -> torch.Tensor:
        """ Make masks, s.t. they have same length """
        mask = self.pad_to_n(mask, n_points=self.n_mask_pts)
        return torch.from_numpy(mask)
    
    @staticmethod
    def pad_to_n(hull: np.ndarray, n_points: int = 65) -> np.ndarray:
        """ Pad hull to always have n_points coordinates """
        # Can't clip points yet
        to_pad = n_points - hull.shape[0]
        assert (to_pad >= 0), f'cannot remove points, len of hull: {len(hull)}'  # Can't remove points yet.
        if to_pad > 0:
            # takes random points from the hull and adds a new point between point n and n+1
            point_indices = np.random.randint(0, hull.shape[0] - 2, size=to_pad)
            p1s = hull[point_indices]
            p2s = hull[point_indices+1]
            new_points = p2s + ((p1s-p2s) / 2)
            hull = np.vstack((hull, new_points))
        return hull


class VehicleImagePsiDataset(VehicleMaskPsiDataset):
    """ Dataset from carla; X: Vehicle image cut from bounding box, y: psi vector """
    def __len__(self):
        return len(self.vehicle_data)

    def __getitem__(self, idx):
        """ 
            Return X, y given idx
            X: 
            y: psi vector (1*2)
        """
        label = self.vehicle_data[idx][0]
        img = self._get_bb_img(idx)
        psi = self._normalize_psi(label['psi'], label['gcp'])
        return img, psi


def bb_to_2d(bb):
    x_min, x_max = np.min(bb[:, 0]), np.max(bb[:, 0])
    y_min, y_max = np.min(bb[:, 1]), np.max(bb[:, 1])
    return x_min, y_min, x_max, y_max

