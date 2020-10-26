from PIL import Image, ImageDraw
import numpy as np
import torch
from torch_mtcnn import detect_faces, show_bboxes
from PIL import Image
import torchvision.transforms as T

image_path = "data\\images\\0001.jpg"
image = Image.open(image_path)
bounding_boxes, landmarks = detect_faces(image)

img1 = ImageDraw.Draw(image)

for i in range(len(bounding_boxes)):
    box = bounding_boxes[i][:4]
    print(box)

    buffer = -11 if box[1] <= 10 else 11
    img1.rectangle(box, outline ="red")
