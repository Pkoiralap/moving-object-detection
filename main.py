from src.finder import findobjects_imagepath, compare_boxes

boxes1 = findobjects_imagepath("data/fifa2.jpeg")
boxes2 = findobjects_imagepath("data/fifa2.jpeg")

result = compare_boxes(boxes1, boxes2)
print(result)