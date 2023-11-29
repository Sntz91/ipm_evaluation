import numpy as np
import cv2

def get_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def points_to_mask(points, image_shape):
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return mask

def calculate_iou(hull_pred, hull_gt, w, h):
    mask_predicted = points_to_mask(hull_pred, (h, w))
    mask_gt = points_to_mask(hull_gt, (h, w))
    iou = get_iou(mask_gt, mask_predicted)
    return iou
