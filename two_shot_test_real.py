import os
import torch
import torchvision
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from data.face_detection_data_loader import face_classes

SAVED_MODEL_DIR = "saved_models"
SAVED_DET_NET = "faceDetectorBest.pt"
SAVED_CLA_NET = "faceClassificationBest.pt"
IMAGES_DIR = "data/darren_test/"
SAVE_OUTPUTS_DIR = "results/model_outputs/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Faster RCNN Model - pretrained on COCO - detection net
detectionNet = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = len(face_classes) # Should be 2 - backround and face
in_features = detectionNet.roi_heads.box_predictor.cls_score.in_features
detectionNet.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Resnet 18 - classification net
classificationNet = torchvision.models.resnet18(pretrained=True)
classificationNet.fc = nn.Linear(512, 4) # Overwrite clasification layer from ImageNet

# Load Models
detectionNet.load_state_dict(torch.load(os.path.join(SAVED_MODEL_DIR, SAVED_DET_NET)))
classificationNet.load_state_dict(torch.load(os.path.join(SAVED_MODEL_DIR, SAVED_CLA_NET)))

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
    
    # Pass entire image through face detetion network
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
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]

        # Pass each bounding box crop through the classification network to get label
        image_crop = image[y1:y2+1, x1:x2+1, :] # H, W, C
        label = torch.argmax(classificationNet(image_crop))

        score = scores[index].numpy()

        font_size = 40
        font = ImageFont.truetype("arial.ttf", font_size)
        img1.rectangle(box, outline ="red")
        img1.text([box[0], box[1]], face_classes[label], font=font)

    #image.show() 
    image.save(SAVE_OUTPUTS_DIR + f"realTest{idx}.jpg")
