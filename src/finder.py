import cv2
import numpy as np

from .yolo.yolo import yolo, draw_names


def findobjects_imagepath(image_path):
    """
    returns the bounding boxes of the detected objects in the image

    Args:
        image_path (str) : the image path

    Returns:
        bboxes (list[list[4]]) : the bounding boxes of objects detected in the image
    """

    image = cv2.imread(image_path)

    # changes image
    bboxes = yolo(image)

    # draw_names(image, bboxes)

    # cv2.imshow("football", image)
    # cv2.waitKey(0)
    # bboxes = []
    return bboxes


def findobjects(image):
    """
    returns the bounding boxes of the detected objects in the image

    Args:
        image_path (np.array) : the image in np array

    Returns:
        bboxes (list[list[4]]) : the bounding boxes of objects detected in the image
    """

    # changes image
    bboxes = yolo(image)
    return bboxes
