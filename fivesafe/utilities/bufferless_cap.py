import cv2 
import queue
import threading
import time
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

# bufferless VideoCapture
class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return True, self.q.get()
  

# TODO
class VideoCapture2:
    def __init__(self, url: str, name: str) -> None:
        self.cap = cv2.VideoCapture(url)
        self.v_writer = self._get_v_writer(name, self.cap)
        self.q = queue.Queue()
        t = threading.Thread(target=self._update)
        t.daemon = True
        t.start()

    @staticmethod
    def _get_v_writer(name: str, capture: cv2.VideoCapture) -> cv2.VideoWriter:
        """
            Given a capture and a name, return a VideoWriter Object. 
        """
        fps = capture.get(cv2.CAP_PROP_FPS)
        size = VideoCapture2.get_frame_size(capture)

        return cv2.VideoWriter(
            f'out/{name}_{time.strftime("%Y-%m-%d-%H:%M:%S", time.gmtime())}.avi', # TODO
            cv2.VideoWriter_fourcc(*'XVID'),
            fps, size
        )

    def _update(self) -> None:
        """
            Endless Loop, refreshing frames. Running in own Thread.
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print('cannot connect')
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self) -> np.array:
        """
            Read latest frame.
        """
        return True, self.q.get()

    @staticmethod
    def get_frame_size(capture: cv2.VideoCapture) -> tuple:
        """
            Return (width, height) of capture object.
        """
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    def isOpened(self) -> bool:
        return self.cap.isOpened()


class MultiCameraCapture:
    def __init__(self, cameras: dict) -> None:
        assert cameras
        self.captures = {}
        for camera_name, url in cameras.items():
            cap = VideoCapture2(url, camera_name)
            assert cap.isOpened(), f"Camera {camera_name} could not be opened."
            self.captures[camera_name] = cap

    def read_frames(self) -> dict:
        """
            Return current frame for each camera;
            Dict with format {"Camera Name": frame,...}
        """
        frames = {}
        for camera_name, cap in self.captures.items():
            frames[camera_name] = self.read_frame(cap)
        return frames

    @staticmethod
    def read_frame(capture: cv2.VideoCapture) -> np.array:
        """
            Get Current frame of given capture.
        """
        ret, frame = capture.read()
        if not ret:
            print("empty frame")
            return
        return frame

    @staticmethod
    def stack_frames(frames: tuple) -> np.array:
        """
            Convert Tuple of frames to one stacked Array. 
        """
        return np.hstack(frames)

    @staticmethod
    def show_combined_frame(frames: dict, fps: float, record_flg: bool) -> int:
        """
            CV-imshow current frames, stacked.
            Additionally, show fps 
        """
        frames = tuple(frames.values())
        frame_stack = MultiCameraCapture.stack_frames(frames)
        MultiCameraCapture._put_fps(frame_stack, fps)
        MultiCameraCapture._put_record_flg(frame_stack, record_flg)
        cv2.imshow('test', frame_stack)

        k = cv2.waitKey(1)
        return k

    @staticmethod
    def _put_record_flg(frame: np.array, flag: bool) -> None:
        if not flag:
            return
        cv2.circle(frame, (12, 12), 8, (0,0,255), -1)
        cv2.putText(frame, "REC", (25,19), FONT, .75, (20,20,20),2)

    @staticmethod
    def _put_fps(frame: np.array, fps: float) -> None:
        """
            Put CV2 Text with FPS onto frame. 
        """
        cv2.rectangle(frame, (1000,0), (1150,40), (255,255,255), -1)
        cv2.putText(frame, str(round(fps,1))+' fps', (1010,30), FONT, .75, (0,0,0),2)