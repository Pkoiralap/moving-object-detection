import cv2
import numpy as np
from tqdm import tqdm

from .finder import findobjects
from .comparer import compare_boxes
from .yolo.yolo import draw_names

def frame_iter(capture, description):
    def _iterator():
        while capture.grab():
            yield capture.retrieve()[1]
    return tqdm(
        _iterator(),
        desc=description,
        total=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    )


def get_new_index_to_name_map(old_index2namemap, objects_map):
    new_index2namemap = {}
    # change index2namemap here
    for i in old_index2namemap:
        new_key = objects_map.get(i)
        if not new_key:
            continue
        new_index2namemap[new_key] = old_index2namemap[i]
    return new_index2namemap


def show_people(video_path, frame_number=0):
    cap = cv2.VideoCapture(video_path)
    i = 0
    frame = None

    for curr_frame in frame_iter(cap, "progress"):
        if i != frame_number:
            continue
        i += 1
        frame = curr_frame
        break

    if frame is None:
        return {}

    boxes = findobjects(frame)
    draw_names(frame, boxes)
    cv2.imshow(f"Frame number: {frame_number}", frame)

    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()


def run(video_path, index2namemap):
    cap = cv2.VideoCapture(video_path)
    # frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    old_boxes = None
    i = -1

    for frame in frame_iter(cap, "progress"):
        i += 1
        if i % 10 != 0:
            continue

        if old_boxes is None:
            old_boxes = findobjects(frame)
            continue
        
        print("old index map", index2namemap)

        new_boxes = findobjects(frame)

        objects_map = compare_boxes(old_boxes, new_boxes)
        index2namemap = get_new_index_to_name_map(index2namemap, objects_map)
        old_boxes = new_boxes

        print(index2namemap, objects_map)
        # set names here
        draw_names(frame, new_boxes, index2namemap)
        cv2.imshow('Video', frame)

        # Hit 'ESC' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()
