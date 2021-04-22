from src.finder import findobjects_imagepath
from src.video import run, show_people

# boxes1 = findobjects_imagepath("data/fifa2.jpeg")
# boxes2 = findobjects_imagepath("data/fifa2.jpeg")
# result = compare_boxes(boxes1, boxes2)
# print(result)

# show_people("data/video.mov")
run("data/video.mov", {10: "Bakra", 9: "Bakra", 8: "Bakra"})
