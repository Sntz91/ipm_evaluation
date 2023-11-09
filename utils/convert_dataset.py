import random 
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

def is_vehicle_out_of_bounds(gcp, w=640, h=480):
    x, y = gcp
    if x > w or x < 0 or y > h or y < 0:
        return True
    return False

def get_vehicles_from_ts(data, ts):
    return data[ts]

def get_vehicles_from_random_ts(data):
    idx_ts = random.choice(list(data.keys()))
    return idx_ts, data[idx_ts]

def filter_vehicles_not_in_img(data, w=1920, h=1080):
    return [vehicle for vehicle in data if not is_vehicle_out_of_bounds(vehicle['gcp'], w, h)]

def get_random_vehicle(data):
    idx = random.randint(0, len(data)-1)
    return data[idx]

def get_vehicles_from_ts(data, ts):
    return data[ts]

def get_vehicle_ids(data):
    return [vehicle['id'] for vehicle in data]

def filter_tv_vehicles(data, vehicle_ids):
    return [vehicle for vehicle in data if vehicle['id'] in vehicle_ids]

def cut_mask(bb, image):
    bb2d = bb_to_2d(np.array(bb))
    xmin, ymin, xmax, ymax = bb2d
    image_cropped = np.asarray(image)[ymin:ymax, xmin:xmax]
    return image_cropped

def get_mask(bb, image_cropped, image_seg):
    rgba, counts = np.unique(image_cropped.reshape(-1,4), axis=0, return_counts=True)
    # Filter street
    #mask = ~np.all(rgba == np.array([1, 94, 110, 255]), axis=1)
    mask = ~np.all((rgba == np.array([1, 61, 111, 255])) | (rgba == np.array([1, 65, 111, 255])), axis=1)
    rgba = rgba[mask]
    counts = counts[mask]
    target_value = rgba[np.argmax(counts)]
    mask = np.all(np.asarray(image_seg) == target_value, axis=-1)
    return mask

def get_hull(mask):
    object_pixels = np.column_stack(np.where(mask))
    hull = cv2.convexHull(object_pixels)
    hull = hull.squeeze()
    return hull

def bb_to_2d(bb):
    x_min, x_max = np.min(bb[:, 0]), np.max(bb[:, 0])
    y_min, y_max = np.min(bb[:, 1]), np.max(bb[:, 1])
    return x_min, y_min, x_max, y_max

def plot_2d_box(bb):
    x_min, y_min, x_max, y_max = bb_to_2d(bb)
    plt.plot([x_min, x_min], [y_min, y_max], color='red')
    plt.plot([x_max, x_max], [y_min, y_max], color='red')
    plt.plot([x_min, x_max], [y_min, y_min], color='red')
    plt.plot([x_min, x_max], [y_max, y_max], color='red')

def plot_3d_box(bb):
    edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]
    for edge in edges:
        plt.plot(bb[edge, 0], bb[edge, 1], color='blue')

def plot_gcp(gcp, size=30):
    _ = plt.scatter(gcp[0], gcp[1], color='cyan', marker='*', s=size)

def plot_psi(gcp, psi):
    x, y = gcp
    x2, y2 = psi
    _ = plt.plot([x, x2], [y, y2], color='red', lw=2)

def plot_hull(hull):
    hull = np.vstack((hull, hull[0, :]))
    plt.plot(hull[:, 1], hull[:, 0], color='orange')

def append_hull(base_url, data):
    fname = base_url+data[0]['img']
    fname_seg = base_url + 'output/seg/' + fname.split('/')[-1]
    image = Image.open(fname)
    image_seg = Image.open(fname_seg)

    for vehicle in data:
        image_cropped = cut_mask(vehicle['bb'], image_seg)
        mask = get_mask(vehicle['bb'], image_cropped, image_seg)
        hull = get_hull(mask)
        vehicle['hull'] = hull
    return data, image_seg

def append_min_area_rect(data):
    for vehicle in data:
        min_area_rect = cv2.minAreaRect(vehicle['hull']) # Maybe need rotation
        box = cv2.boxPoints(min_area_rect)
        box = np.intp(box)
        vehicle['min_area_rect'] = box
    return data

def plot_min_area_rect(box):
    box = np.vstack((box, box[0, :]))
    plt.plot(box[:, 1], box[:, 0], color='cyan')

def get_data(ts, base_url='./', w=640, h=480):
    # Load Pickle Data
    with open(base_url + 'output/pv/data.pickle', 'rb') as handle:
        data_pv = pickle.load(handle)

    with open(base_url + 'output/tv/data.pickle', 'rb') as handle:
        data_tv = pickle.load(handle)

    #print(data_pv) 
    for t, vehicles_t in data_pv.items():
        vehicles_t = filter_vehicles_not_in_img(vehicles_t)
        if vehicles_t == []:
            continue
        vehicles_t, _ = append_hull(base_url, vehicles_t)
        vehicles_t = append_min_area_rect(vehicles_t)

    for t, vehicles_t in data_tv.items():
        vehicles_t = filter_vehicles_not_in_img(vehicles_t)
        if vehicles_t == []:
            continue
        #vehicles_t, _ = append_hull(base_url, vehicles_t)
        #vehicles_t = append_min_area_rect(vehicles_t)

    with open(base_url + 'output/pv/data_new.pickle', 'wb') as handle:
        pickle.dump(data_pv, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(base_url + 'output/tv/data_new.pickle', 'wb') as handle:
        pickle.dump(data_tv, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Perspective View
    # Perspective View
    #vehicles_at_ts = get_vehicles_from_ts(data_pv, ts)
    #vehicles_at_ts_filtered = filter_vehicles_not_in_img(vehicles_at_ts, w, h)
    #vehicles_pv, image_seg = append_hull(base_url, vehicles_at_ts_filtered)
    #vehicles_pv = append_min_area_rect(vehicles_at_ts_filtered)
    #fname_pv = base_url + vehicles_at_ts_filtered[0]['img']
    #image_pv = Image.open(fname_pv)
    
    # Top View
    #vehicles_at_ts_tv = get_vehicles_from_ts(data_tv, ts)
    #v_ids = get_vehicle_ids(vehicles_pv)
    #vehicles_tv = filter_tv_vehicles(vehicles_at_ts_tv, v_ids)
    #fname_tv = base_url+vehicles_tv[0]['img']
    #image_tv = Image.open(fname_tv)

    #print(data_pv)
    
    #return image_pv, image_tv, image_seg, vehicles_pv, vehicles_tv

#vehicles_pv = {}
#image_pv, image_tv, image_set, vehicles_pv_t, vehicles_tv_t = get_data(5, base_url='/Users/tobias/ziegleto/data/5Safe/carla/circle/', w=1920, h=1080)
#print(vehicles_pv)

get_data(5, base_url='/Users/tobias/ziegleto/data/5Safe/carla/circle/', w=1920, h=1080)
