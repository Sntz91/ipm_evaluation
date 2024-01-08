from torch.utils.data import Dataset
from ...utilities import Dict2ObjParser
import yaml
import json
import cv2
import os

class Dataset(Dataset):
    def __init__(self, top_view_cfg, perspective_views_cfg):
        self.top_view_cfg = top_view_cfg
        self.perspective_view_cfg = perspective_views_cfg

    def __len__(self):
        pass

    def get_image_size(self):
        return 1002, 1920, 3 # TODO
    
    @staticmethod
    def get_gcps(objects: dict):
        gcps = []
        for id, obj in objects.items():
            gcps.append([obj['x'], obj['y']])
        return gcps

    def __getitem__(self, idx):
        """ 
        Returns 
        - Tuple of Images for Perspective- and Top View  
        - Tuple of Labels for Perspective- and Top View
        Note: Perspective Views can be > 1 and must be indexed
        """
        idx += 1
        # Top View
        tv_img_url = os.path.join(
            self.top_view_cfg.input_dir, 
            f'{idx:05}.jpg'
        )
        img_tv = cv2.imread(tv_img_url)

        with open(self.top_view_cfg.labels, 'r') as f:
            labels_tv = json.load(f)
            labels_tv = labels_tv[str(idx)]

        # Perspective Views
        imgs_pv = []
        labels_pv = []
        for cfg in self.perspective_view_cfg:
            pv_img_url = os.path.join(
                cfg.input_dir, 
                f'{idx+cfg.offset_to_top_view:05}.jpg'
            )
            imgs_pv.append(cv2.imread(pv_img_url))

            with open(cfg.labels, 'r') as f:
                labels = json.load(f)
                idx_ = idx+cfg.offset_to_top_view
                labels_pv.append(labels[str(idx_)])

        return (imgs_pv, img_tv), (labels_pv, labels_tv)






if __name__ == '__main__':
    cfg_name = './conf/vup1_dataset.yaml'
    with open (cfg_name, 'r') as file:
        cfg = yaml.safe_load(file)
        cfg = Dict2ObjParser(cfg).parse()

    dataset = Dataset(cfg.dataset.top_view_cfg, cfg.dataset.perspective_view_cfg)
    (imgs_pv, img_tv), (labels_pv, labels_tv) = dataset.__getitem__(200)

    img_pv = imgs_pv[0]
    img_pv2 = imgs_pv[1]
    labels_pv1 = labels_pv[0]
    labels_pv2 = labels_pv[1]

    # Plotting
    cv2.namedWindow("img_pv", cv2.WINDOW_NORMAL)
    cv2.namedWindow("img_pv2", cv2.WINDOW_NORMAL)
    cv2.namedWindow("img_tv", cv2.WINDOW_NORMAL)

    height, width, _ = img_pv.shape
    for obj_id, obj_values in labels_pv2.items():
        print(obj_values)
        x, y, w, h  = obj_values["x"], obj_values["y"], obj_values["width"], obj_values["height"] 
        cv2.circle(img_pv2, (int(x), int(y)),5, (255, 255, 0), -1)
        cv2.circle(img_pv2, (int(x + w), int(y + h)),5, (255, 255, 0), -1)

    height, width, _ = img_tv.shape
    for obj_id, obj_values in labels_tv.items():
        x, y, w, h  = obj_values["x"], obj_values["y"], obj_values["width"], obj_values["height"] 
        cv2.circle(img_tv, (int(x), int(y)),5, (255, 255, 0), -1)
        cv2.circle(img_tv, (int(x + w), int(y + h)),5, (255, 255, 0), -1)

    cv2.imshow('img_pv', img_pv)
    cv2.imshow('img_pv2', img_pv2)
    cv2.imshow('img_tv', img_tv)
    cv2.waitKey(0)
    