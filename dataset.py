from torch.utils.data import Dataset
import numpy as np
import pickle
import cv2
import torch


class VehicleDataset(Dataset):
    def __init__(self, base_url, transform=None, target_transform=None):
        # read annotation file
        self.base_url = base_url
        self.transform = transform
        self.target_transform = target_transform

        with open(base_url + 'output/pv/data_new.pickle', 'rb') as handle:
            self.data_pv = pickle.load(handle)

        with open(base_url + 'output/tv/data_new.pickle', 'rb') as handle:
            self.data_tv = pickle.load(handle)

        assert len(self.data_pv) == len(self.data_tv), 'tv pv not same length?'

    def __len__(self):
        return len(self.data_pv)-1

    def __getitem__(self, idx):
        idx = idx+1 
        label_pv = self.data_pv[idx]
        label_tv = self.data_tv[idx]
        image_pv_url = self.base_url + label_pv[0]['img']
        image_tv_url = self.base_url + label_tv[0]['img']
        image_pv = cv2.imread(image_pv_url)
        image_tv = cv2.imread(image_tv_url)

        return (image_pv, image_tv), (label_pv, label_tv) 


class VehiclePSIDataset(Dataset):
    def __init__(self, base_url):
        # read annotation file
        self.base_url = base_url

        # Load to RAM
        with open(base_url + 'output/pv/data_new.pickle', 'rb') as handle:
            self.data_pv = pickle.load(handle)

    def __len__(self):
        return len(self.data_pv)-1

    def __getitem__(self, idx): # Note this only works for 1 car per image
        label_t1 = self.data_pv[idx+2][0]
        label_t0 = self.data_pv[idx+1][0]

        mask_t1 = self._normalize_mask(label_t1['hull'])
        mask_t0 = self._normalize_mask(label_t0['hull'])
        psi = self._normalize_psi(label_t1['psi'], label_t1['gcp'])
        # nr_of_masks, mask_len, point_dimension
        masks = torch.stack((mask_t0, mask_t1))
        return masks, psi
    
    def _get_image(self, idx):
        label_pv = self.data_pv[idx+2]
        image_pv_url = self.base_url + label_pv[0]['img']
        image_pv = cv2.imread(image_pv_url)
        return image_pv
    
    def _normalize_psi(self, psi, gcp):
        # Normalize to 0
        psi_normalized = [c - c0 for c, c0 in zip(gcp, psi)]
        # Normalize length to be 1
        psi_normalized = psi_normalized / np.sqrt(np.dot(psi_normalized, psi_normalized))
        return torch.Tensor(-psi_normalized)

    def _normalize_mask(self, mask):
        mask = self.pad_to_n(mask)
        return torch.from_numpy(mask)
    
    @staticmethod
    def pad_to_n(hull: np.ndarray, n_points: int = 50) -> np.ndarray:
        """Pad hull to always have n_points coordinates
 
        Args:
            hull (np.ndarray): shape: [n, 2]
            n_points (int, optional): _description_. Defaults to 32.
 
        Returns:
            npt.NDArray: Padded hull
        """
        # Pads the convex hull to always have 32 points
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

def bb_to_2d(bb):
    x_min, x_max = np.min(bb[:, 0]), np.max(bb[:, 0])
    y_min, y_max = np.min(bb[:, 1]), np.max(bb[:, 1])
    return x_min, y_min, x_max, y_max

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    dataset = VehiclePSIDataset(base_url='/Users/tobias/ziegleto/data/5Safe/carla/circle/')

    for i in range(len(dataset)):
        hull, psi = dataset.__getitem__(i)
        img = dataset._get_image(i)
        plt.imshow(img)
        hull_ = hull[0].numpy()
        plt.scatter(hull_[:, 1], hull_[:, 0])
        plt.show()
        pt0 = (500, 500)
        img = cv2.arrowedLine(img, pt0, (int(pt0[0]+psi[0]*100), int(pt0[1]+psi[1]*100)), (0, 255, 0), 5)
        img = cv2.imshow('img', img)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        print(psi)

        # PLOT MASKS TODO

