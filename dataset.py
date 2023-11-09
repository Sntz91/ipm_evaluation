from torch.utils.data import Dataset
import numpy as np
import pickle
import cv2


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


def bb_to_2d(bb):
    x_min, x_max = np.min(bb[:, 0]), np.max(bb[:, 0])
    y_min, y_max = np.min(bb[:, 1]), np.max(bb[:, 1])
    return x_min, y_min, x_max, y_max

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = VehicleDataset(base_url='/Users/tobias/ziegleto/data/5Safe/carla/circle/')
    (image_pv, image_tv), (label_pv, label_tv) = dataset.__getitem__(2)

#train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
#train_features, train_labels = next(iter(train_dataloader))

#print(train_labels[0])
