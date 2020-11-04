import os
import torch
import torchvision
import torchvision.transforms as T
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt

# from data.face_detection_data_loader import face_classes
from data.face_classification_data_loader import face_classes


SAVED_MODEL_DIR = "saved_models"
SAVED_DET_NET = "faceDetectorBest.pt"
SAVED_CLA_NET = "Face_Classification_ADAM_StepLR_25_5e-05_best.pt"
IMAGES_DIR = "data/real_test_cases/"
SAVE_OUTPUTS_DIR = "results/model_outputs/"

if not os.path.exists(os.path.join(SAVE_OUTPUTS_DIR,"TWO_SHOT_TEST_RONEN_AGAIN")):
    os.makedirs(os.path.join(SAVE_OUTPUTS_DIR,"TWO_SHOT_TEST_RONEN_AGAIN"))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Faster RCNN Model - pretrained on COCO - detection net
detectionNet = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# num_classes = len(face_classes) # Should be 2 - backround and face
num_classes = 5
in_features = detectionNet.roi_heads.box_predictor.cls_score.in_features
detectionNet.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Resnet 18 - classification net
classificationNet = torchvision.models.resnet18(pretrained=True)
classificationNet.fc = nn.Linear(512, 4) # Overwrite clasification layer from ImageNet

# Load Models
detectionNet.load_state_dict(torch.load(os.path.join(SAVED_MODEL_DIR, SAVED_DET_NET)))
classificationNet.load_state_dict(torch.load(os.path.join(SAVED_MODEL_DIR, SAVED_CLA_NET)))
detectionNet.eval().to(device)
classificationNet.eval().to(device)

# Load Test Data
image_names = os.listdir(IMAGES_DIR)

# ClassificationNet Transforms
transforms = T.Compose([
    T.ToPILImage(),
    T.Resize((128, 128)),
    T.ToTensor()
])
with torch.no_grad():
    # Get outputs of model
    # model.to(device).eval()
    ctr = 0
    for idx in range(len(image_names)):
        image_path = os.path.join(IMAGES_DIR, image_names[idx])
        image = Image.open(image_path)
        image = np.array(image) / 255.0
        image = torch.tensor(image, dtype=torch.float).permute(2,0,1).contiguous()
        image = torch.unsqueeze(image, dim=0)

        # Pass entire image through face detetion network
        out = detectionNet(image.to(device))

        boxes = out[0]['boxes'].cpu().detach()
        labels = out[0]['labels'].cpu().detach()
        scores = out[0]['scores'].cpu().detach()
        keep = torchvision.ops.nms(boxes=boxes, scores=scores, iou_threshold=0.3)

        image_to_draw = image[0]
        image_to_draw = np.moveaxis(image_to_draw.numpy(),0,2)
        image_to_draw = 255 * image_to_draw
        image_to_draw = image_to_draw.astype(np.uint8)
        image_to_draw = T.ToPILImage()(image_to_draw)
        img1 = ImageDraw.Draw(image_to_draw)

        for i in range(len(keep)):
            index = keep[i]
            box = boxes[index].numpy()
            x1 = int(round(box[0]))
            y1 = int(round(box[1]))
            x2 = int(round(box[2]))
            y2 = int(round(box[3]))


            # Pass each bounding box crop through the classification network to get label
            # image = torch.squeeze(image, dim=0)

            image_crop = image[:, :, y1:y2+1, x1:x2+1] # B, C, H, W
            image_crop = image_crop[0, :, :, :]


            ####################
            # image_crop_to_draw = np.moveaxis(image_crop.numpy(), 0, 2)
            # image_crop_to_draw = 255 * image_crop_to_draw
            # image_crop_to_draw = image_crop_to_draw.astype(np.uint8)
            # image_crop_to_draw = T.ToPILImage()(image_crop_to_draw)
            # plt.imshow(image_crop_to_draw)
            # ctr += 1
            # plt.savefig(os.path.join(SAVE_OUTPUTS_DIR,"TWO_SHOT_TEST_RONEN_AGAIN", f"single_face_{ctr}.jpg"))
            # plt.show()
            ####################

            image_crop = transforms(image_crop)
            image_crop = torch.unsqueeze(image_crop, dim=0).to(device)
            label = torch.argmax(classificationNet(image_crop))

            score = scores[index].numpy()

            font_size = 60
            font = ImageFont.truetype("arial.ttf", font_size)

            img1.rectangle(box, outline ="red", width=2)
            img1.text([box[0], box[1]], face_classes[label], font=font)

        # image_to_draw.show()
        image_to_draw.save(os.path.join(SAVE_OUTPUTS_DIR,"TWO_SHOT_TEST_RONEN_AGAIN", f"realTest{idx}.jpg"))
