from ..utilities.colors import COLORS
from ..object_detection import draw_detection_offset

def draw_track(frame, track, draw_detection_id=True, color=None):
    # Pick Color 
    if not color:
        color = COLORS[track.id] 
    # Draw Track
    frame = track.draw_rectangle(
        frame, 
        color=color, 
        thickness=2
    )
    if draw_detection_id:
        frame = track.draw_detection_id(
            frame, 
            offset=(0, -10), 
            color=color
        )
    #frame = track.draw_label(
    #    frame, 
    #    color=color
    #)
    frame = track.draw_id(
        frame, 
        color=color
    )
    frame = track.draw_midpoint(
        frame, 
        color=color
    )
    return frame

def draw_tracks(frame, tracks, colors, draw_detection_id=True):
    for track in tracks:
        color_ = getattr(colors, track.label())
        frame = draw_track(frame, track, draw_detection_id=draw_detection_id, color=color_)
    return frame

def draw_debug_tracks(frame, tracks, detections):
    print(tracks)
    for track, detection in zip(tracks, detections):
        draw_track(frame, track)
        draw_detection_offset(frame, detection)
    return frame