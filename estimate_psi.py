from dataset import VehiclePSIDataset, bb_to_2d
from torch.utils.data import DataLoader
import cv2

dataset = VehiclePSIDataset(base_url='/Users/tobias/ziegleto/data/5Safe/carla/circle/')

train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))

# batch_size, masks, points, [x,y]