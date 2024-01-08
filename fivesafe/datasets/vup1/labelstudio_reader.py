import json
import cv2

def generate_interpolated_entitydict_from_labelstudio_json_export(inputpath, index):
  print("Generating interpolated Data...")
  with open(inputpath, 'r') as f:
    data = json.load(f)

  annotations = data[index]["annotations"][0]["result"]

  keyframedict = dict()

  keyframedict["metainformation"] = {"basefile_for_annotation": data[0]["data"]["video"].rsplit("%5C")[-1],
                                    "frames_total": data[0]["annotations"][0]["result"][0]["value"]["framesCount"],
                                    "duration_total": data[0]["annotations"][0]["result"][0]["value"]["duration"]}
  keyframedict["objects"] = dict()

  for annotation in annotations:
      entitydict =  dict()

      entitydict["class"] = annotation["value"]["labels"][0]
      interpolated_objectdict = interpolate_objectdict_in_relevant_time_interval(annotation["value"]["sequence"])
      entitydict["keyframe_annotations"] = interpolated_objectdict

      keyframedict["objects"][annotation["id"]] = entitydict
  return keyframedict

def interpolation_consistency_check(objectdict_list, interpolated_objectdict_list):
  frame_end = objectdict_list[-1]["frame"]

  if frame_end < len(interpolated_objectdict_list):
    raise Exception("Interpolation Consistency Check Error!")

def interpolate_objectdict_in_relevant_time_interval(objectdict_list):
  if len(objectdict_list) < 2:
    raise Exception("List of Object has only one entry and is too small for Interpolation! Check outputformat in Labeling Tool and close Trajectory!")
  interpolated_objectdict_list = list()
  for i in range(len(objectdict_list)-1):
    curr = objectdict_list[i]
    next = objectdict_list[i+1]

    x_diff = curr["x"] - next["x"]
    y_diff = curr["y"] - next["y"]
    w_diff = curr["width"] - next["width"]
    h_diff = curr["height"] - next["height"]
    rotation_diff = curr["x"] - next["x"]
    frame_diff = abs(curr["frame"] - next["frame"])
    enabled = curr["enabled"]

    interpolator_x = x_diff / frame_diff
    interpolator_y = y_diff / frame_diff
    interpolator_w = w_diff / frame_diff
    interpolator_h = h_diff / frame_diff
    interpolator_rotation = rotation_diff / frame_diff

    # Interpolate items between first and last item
    for j in range(frame_diff):
        new_frame = curr["frame"] + j
        new_x = curr["x"] + j * interpolator_x
        new_y = curr["y"] + j * interpolator_y
        new_w = curr["width"] + j * interpolator_w
        new_h = curr["height"] + j * interpolator_h
        new_rotation = curr["rotation"] + j * interpolator_rotation
        objdict = {"frame": new_frame, 
                   "x": new_x, 
                   "y": new_y, 
                   "width": new_w, 
                   "height": new_h, 
                   "rotation": new_rotation, 
                   "enabled": enabled}
        interpolated_objectdict_list.append(objdict)
    
    # Add last item if end is reached
    if next == objectdict_list[-1]:
      interpolated_objectdict_list.append(next)
  interpolation_consistency_check(objectdict_list, interpolated_objectdict_list)
  return interpolated_objectdict_list
   
def generate_objectdict_for_relevant_timestep(relevant_framenumber, objects):
  objectdict = dict()
  for key, value in objects.items():
    obj_uuid = key
    for elem in value["keyframe_annotations"]:
      if elem["frame"] == relevant_framenumber:
        objectdict[obj_uuid] = {"class": value["class"],
                                "x": elem["x"], 
                                "y": elem["y"], 
                                "width": elem["width"], 
                                "height": elem["height"], 
                                "rotation": elem["rotation"],
                                "enabled": elem["enabled"]}
        break
  return objectdict

   
def generate_timedict_from_interpolated_keyframedict(keyframedict):
  print("Generating Time Series Dict from Labeldata...")
  timedict = dict()
  timeseriesdict = dict()
  framecounter = 1
  while framecounter <= keyframedict["metainformation"]["frames_total"]:
     timeseriesdict[framecounter] = generate_objectdict_for_relevant_timestep(framecounter, keyframedict["objects"])
     framecounter += 1
  timedict["metainformation"] = keyframedict["metainformation"]
  timedict["timeseries"] = timeseriesdict
  return timedict

def generate_timedict_from_labelstudio_export(inputpath, idx):
  keyframedict = generate_interpolated_entitydict_from_labelstudio_json_export(inputpath, idx)
  timedict = generate_timedict_from_interpolated_keyframedict(keyframedict)

  return timedict

if __name__ == "__main__":
  inputpath = "./cam2labels_and_topview_labels.json"

  timedict_pv = generate_timedict_from_labelstudio_export(inputpath, 0)
  timedict_tv = generate_timedict_from_labelstudio_export(inputpath, 1)
  print("Done.")
  #print(timedict)

  frame = 200
  offset = 160-81  # FIND OFFSET MANUALLY
  # MASTER TOPVIEW

  # TODO: Note must be in between 2xx and 8xx
  img_pv = cv2.imread(f"/home/ziegleto/ziegleto/data/5Safe/vup/Pedestrian/processed/camera2/imgs/00{frame}.jpg") 
  img_tv = cv2.imread(f"/home/ziegleto/ziegleto/data/5Safe/vup/Pedestrian/processed/top/imgs/00{frame-offset}.jpg")

  height, width, _ = img_tv.shape
  cv2.namedWindow("img pv", cv2.WINDOW_NORMAL)
  cv2.namedWindow("img tv", cv2.WINDOW_NORMAL)
  print("Meainformation: ", timedict_pv["metainformation"])
  print("Meainformation: ", timedict_tv["metainformation"])
  obj_to_plot = timedict_tv["timeseries"][frame-offset] # 160 - 81
  for obj_uuid, obj_values in obj_to_plot.items():
    x = obj_values["x"] * width/100
    y = obj_values["y"] * height/100
    w = obj_values["width"] * width/100
    h = obj_values["height"] * height/100
  cv2.circle(img_tv, (int(x), int(y)),5, (255, 255, 0), -1)
  cv2.circle(img_tv, (int(x + w), int(y + h)),5, (255, 255, 0), -1)

  height, width, _ = img_pv.shape
  obj_to_plot = timedict_pv["timeseries"][frame] # 160 - 81
  for obj_uuid, obj_values in obj_to_plot.items():
    x = obj_values["x"] * width/100
    y = obj_values["y"] * height/100
    w = obj_values["width"] * width/100
    h = obj_values["height"] * height/100
  cv2.circle(img_pv, (int(x), int(y)),5, (255, 255, 0), -1)
  cv2.circle(img_pv, (int(x + w), int(y + h)),5, (255, 255, 0), -1)

  cv2.imshow("img pv", img_tv)
  cv2.imshow("img tv", img_pv)
  cv2.waitKey(0)

