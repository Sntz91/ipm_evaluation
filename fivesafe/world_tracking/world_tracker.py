import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag
from scipy.spatial.distance import cdist

def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def abs_dist_batch(detections, trackers):
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
  trackers = trackers[:,0:2]
  detections = detections[:,0:2]
  
  dist_array = cdist(detections, trackers, "euclidean")
                                           
  return(dist_array)

def cosine_rvec_smooting(rvec_curr, rvec_candidate, weightfactor = 1):
  if rvec_curr.any() is None:
    return rvec_candidate
  if np.isnan(rvec_candidate).any():
    return rvec_curr
  else:
    dot_product = np.dot(np.squeeze(rvec_curr), np.squeeze(rvec_candidate))
    rvec_delta_angle = np.arccos(dot_product)
    norm_count = 1
    while abs(rvec_delta_angle) > (np.pi):
      if rvec_delta_angle > np.pi:
        rvec_delta_angle = rvec_delta_angle -(np.pi)
        norm_count += 1
      if rvec_delta_angle < np.pi:
        rvec_delta_angle = rvec_delta_angle +(np.pi)
        norm_count += 1
    if dot_product < 0.1 and dot_product > 0:
      rvec_aggergated = rvec_curr + rvec_candidate * weightfactor
    else:
      rvec_aggergated = rvec_curr + np.cos(rvec_delta_angle) * rvec_candidate * weightfactor
      #rvec_aggergated = rvec_candidate 
    if np.linalg.norm(rvec_aggergated) < 0.01:
      rvec  = rvec_curr
    else:
      rvec = rvec_aggergated / np.linalg.norm(rvec_aggergated)

    if dot_product <0:
      rvec = rvec * -1
    return rvec  


class KalmanWorldTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox, dt):
    """
    Initialises a tracker using initial bounding box.
    """
    self.dt = dt
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=4, dim_z=2) 
    # [x, y, vx, vy]
    self.kf.F = np.array([[1,0, self.dt, 0],
                          [0,1,0, self.dt],
                          [0,0,1,0],
                          [0,0,0,1]])
    
    self.kf.H = np.array([[1,0,0,0],
                          [0,1,0,0]])
    R_std = 10
    Q_std = .3
    self.kf.R = np.eye(2) * R_std**2
    q = Q_discrete_white_noise(dim=2, dt=self.dt, var=Q_std**2)
    self.kf.Q = block_diag(q, q)
    self.kf.alpha = 1.2

    self.kf.x = np.array([[bbox[0], bbox[1], 0, 0]]).T
    self.time_since_update = 0
    self.id = KalmanWorldTracker.count
    KalmanWorldTracker.count += 1
    self.history = []
    self.rvec_ringbuffer = []
    self.position_last_rvec_update = np.array([[bbox[0]], [bbox[1]]])
    self.actual_rvec  = np.array([[None], [None]])
    self.tracked_class_id = bbox[2]
    self.detection_id = bbox[3]
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

    self.get_first_rvec_estimate_from_infrastructure()

    #self.original_id = bbox[5] # <--- add to keep track of original IDs

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    #self.original_id = bbox[5]  # <--- update to keep track of original IDs
    self.kf.update((bbox[0:2]))
    self.tracked_class_id = bbox[2]

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(self.kf.x)
    if self.tracked_class_id ==2:
      current_postion = np.array([self.history[0][0], self.history[0][1]])
      ### Marcels Codeblock for Angle Estimation from current Movement Pattern ###
      dist_rvec_update = np.sqrt((self.position_last_rvec_update[0] - current_postion[0]) * (self.position_last_rvec_update[0] - current_postion[0]) + (self.position_last_rvec_update[1] - current_postion[1]) * (self.position_last_rvec_update[1] - current_postion[1]))
          #TODO: Update rvec estimate only if object moved a certain amount
          #TODO: Initial rvec Estimate from Map
      th = 1 *  ((632.26/13.5))
      if dist_rvec_update > th:
        rvec_candidate = self.get_rvec_from_two_points(current_postion, self.position_last_rvec_update)
        self.actual_rvec  = cosine_rvec_smooting(self.actual_rvec, rvec_candidate)
        self.position_last_rvec_update = current_postion

    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return self.kf.x[0:2]
  
  def get_age(self):
    """
    Returns the age of the KalmanWorldTracker
    """
    return self.age
  
  def get_rvec_estimate_from_track_history(self):
    pos_latest = self.rvec_ringbuffer[-1]
    pos_old = self.rvec_ringbuffer[0]
    rvec = self.get_rvec_from_two_points(pos_latest, pos_old)

    return rvec
  
  def get_first_rvec_estimate_from_infrastructure(self):
    if self.tracked_class_id ==2:
      # Initial rvec part
      initial_rvec_positions = [[2400, 1200, -.5, -.5],
                                [1500, 500, .5, 0.5],
                                [2300, 600, -.5, .5],
                                [1200, 1100, .9, 0.1]]
      erglist = list()
      for vec in initial_rvec_positions:
        dist = np.sqrt((vec[0] - self.position_last_rvec_update[0]) * (vec[0] - self.position_last_rvec_update[0]) + (vec[1] - self.position_last_rvec_update[1]) * (vec[1] - self.position_last_rvec_update[1]))
        erglist.append(dist)

        idx = erglist.index(min(erglist))
        self.actual_rvec  = np.array([[initial_rvec_positions[idx][2]], [initial_rvec_positions[idx][3]]])
    else:
      self.actual_rvec = np.array([[None], [None]])

  @staticmethod
  def get_rvec_from_two_points(point1, point2):
    rvec = point2 - point1
    norm = np.linalg.norm(rvec)
    if norm > 0.0001:
      rvec = np.array([(point1[0] - point2[0]) / norm,
                                            (point1[1] - point2[1]) / norm], dtype=np.float32)
    else:
      return np.array([[np.nan], [np.nan]])
    return rvec

    

def associate_detections_to_trackers(detections,trackers, dist_threshold):
  """
  Assigns detections to tracked object (both represented as World Positions)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,4),dtype=int)

  dist_matrix = abs_dist_batch(detections, trackers)

  if min(dist_matrix.shape) > 0:
    a = (dist_matrix < dist_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(dist_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low Distance
  matches = []
  for m in matched_indices:
    if(dist_matrix[m[0], m[1]]>dist_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class WorldSort(object):
  def __init__(self, max_age=5, min_hits=0, dist_threshold = 632.26/13.5*5, dt_std = 1/30):
    """
    Sets key parameters for SORT
    """
    self.dt_std = dt_std
    self.max_age = max_age
    self.min_hits = min_hits
    self.dist_threshhold = dist_threshold
    self.trackers = []
    self.frame_count = 0

  def update(self, dets=np.empty((0, 2))):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 4))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()
      trk[:] = [pos[0][0], pos[1][0], pos[2][0], pos[3][0]]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.dist_threshhold)

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanWorldTracker(dets[i,:], self.dt_std)
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()
        if (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):  #(trk.time_since_update < 1) and
          tmp = np.concatenate((d[0], d[1], [trk.id+1], trk.actual_rvec[0], trk.actual_rvec[1], [trk.detection_id], [trk.tracked_class_id]))
          ret.append(tmp.reshape(1,-1)) # +1 as MOT benchmark requires positive  # <--- add [trk.original_id] to the returned set
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))