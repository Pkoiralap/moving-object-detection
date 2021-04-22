import argparse
import cv2
import numpy as np


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def draw_prediction(img, classes, class_id, confidence, x, y, x_plus_w, y_plus_h):
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    return color


def draw_name(img, boxes):
    for box in boxes:
        x = box["x"]
        y = box["y"]
        color = box["color"]
        label = box["class"]
        cv2.putText(img, f"{box['index']}-{label}", (round(x) - 10, round(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def get_bbox_average(img_rgb, x1, y1, x2, y2, row_number=40):
    box = img_rgb[y1:y2, x1:x2]
    box = box.reshape(3, box.shape[0] * box.shape[1])
    box = box[:, :(box[0].size // row_number) * row_number]
    box = box.reshape(row_number, box.size//row_number)
    mean_list = np.average(box, axis=1)
    return mean_list


def yolo(image, classes="src/yolo/classes.txt", config="src/yolo/yolo.cfg", weights="src/yolo/yolov3.weights"):
    """
    find objects using yolo from the image

    Args:
        image (np.array) : the input image
        config (str) : path to yolo config file
        weights (str) : path to the weights
        classes (list) : list of objects to detect

    Returns:
    """

    with open(classes, 'r') as in_file:
        classes = [line.strip() for line in in_file.readlines()]

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    net = cv2.dnn.readNet(weights, config)

    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #convert it to RGB channel
    return_val = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        color = draw_prediction(image, classes, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        average = get_bbox_average(img_rgb, round(x), round(y), round(x+w), round(y+h))

        return_val.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "class": classes[class_ids[i]],
            "confidence": confidences[i],
            "color": color,
            "centroid": np.array([x + w/2, y + h/2]),
            "average": average,
            "index": i
        })

    return return_val
