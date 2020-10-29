import os
import torch
import torchvision
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from data.face_detection_data_loader import face_classes

SAVED_MODEL_DIR = "saved_models"
SAVED_MODEL_FINAL_NAME = "finalModel.pt"
SAVED_MODEL_BEST_NAME = "bestModel.pt"
IMAGES_DIR = "data/darren_test/"
SAVE_OUTPUTS_DIR = "results/model_outputs/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Faster RCNN Model - pretrained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = len(face_classes)

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Load Model
model.load_state_dict(torch.load(os.path.join(SAVED_MODEL_DIR, SAVED_MODEL_BEST_NAME)))

# Load Test Data
image_names = os.listdir(IMAGES_DIR)

# Get outputs of model
model.to(device).eval()
for idx in range(len(image_names)):
    image_path = os.path.join(IMAGES_DIR, image_names[idx])
    image = Image.open(image_path)
    image = np.array(image) / 255.0
    image = torch.tensor(image, dtype=torch.float).permute(2,0,1).contiguous()
    image = torch.unsqueeze(image, dim=0)
    
    out = model(image.to(device))

    boxes = out[0]['boxes'].cpu().detach()
    labels = out[0]['labels'].cpu().detach()
    scores = out[0]['scores'].cpu().detach()

    keep = torchvision.ops.nms(boxes=boxes, scores=scores, iou_threshold=0.3)
    
    image = image[0]
    image = np.moveaxis(image.numpy(),0,2)
    image = 255 * image
    image = image.astype(np.uint8)
    image = T.ToPILImage()(image)
    img1 = ImageDraw.Draw(image)

    for i in range(len(keep)):
        index = keep[i]

        box = boxes[index].numpy()
        label = labels[index].numpy()
        score = scores[index].numpy()

        font_size = 40
        font = ImageFont.truetype("arial.ttf", font_size)
        img1.rectangle(box, outline ="red")
        img1.text([box[0], box[1]], face_classes[label], font=font)

    #image.show() 
    image.save(SAVE_OUTPUTS_DIR + f"realTest{idx}.jpg")
