import cv2
import numpy as np

from .yolo.yolo import yolo, draw_name


def mean_square_difference(arr1, list_arr2):
    """
    [summary]

    Args:
        arr1 (np.array) : [description]
        list_arr2 (list[np.array]) : [description]

    Returns:
        [type] : [description]
    """
    return np.array([(np.square(arr1 - arr2)).mean() for arr2 in list_arr2])


def calculate_similarities(box1, boxes2, indexes, similarity_metrices):
    res = np.ones((1, len(indexes)))
    for metric in similarity_metrices:
        sim = mean_square_difference(box1[metric], [boxes2[index][metric] for index in indexes])
        res = np.multiply(res, sim)
    return res


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

    # draw_name(image, bboxes)

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


def compare_boxes(boxes1, boxes2):
    """
    [summary]

    Args:
        box1 ([type]) : [description]
        box2 ([type]) : [description]

    Exceptions:

    Yields:

    Returns:
    """

    index_map = {}
    for index1, box1 in enumerate(boxes1):
        # print(index1, box1)
        indexes = range(max(index1 - 3, 0), min(index1 + 3, len(boxes2)-1) + 1)
        similarity_metrices = ["average", "centroid"]
        sim_list = calculate_similarities(box1, boxes2, indexes, similarity_metrices)

        most_similar_index = np.argmin(sim_list)
        index_map[index1] = most_similar_index + indexes[0]

    return index_map
