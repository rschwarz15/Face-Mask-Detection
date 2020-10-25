import os
from PIL import ImageDraw
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
import time

from util import nms
from data.data_loader import train_data_loader, test_data_loader, face_classes

SAVED_MODEL_DIR = "saved_models"
SAVED_MODEL_NAME = "bestModel.pt"
EPOCHS = 30
LEARNING_RATE = 1e-3
TRAIN = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test():
    total_epoch_loss = 0
        
    for images, boxes, labels in test_data_loader:
        #Loading images & targets on device
        targets = []
        for i in range(len(boxes)):
            targets.append({'boxes': boxes[i].to(device), 'labels': labels[i].to(device)})
        #targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        #Forward propagation
        out = model(images.to(device), targets)
        losses = sum(loss for loss in out.values())

        #Average loss
        loss_value = losses.item()
        total_epoch_loss += loss_value

    epoch_train_loss = total_epoch_loss / len(test_data_loader)

    return epoch_train_loss


def train(epochs):
    train_loss_logger = []
    test_loss_logger = []
    best_test_loss = 999

    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        total_epoch_loss = 0
        new_best = ""
        
        #Retriving Mini-batch
        for images, boxes, labels in train_data_loader:

            #Loading images & targets on device
            targets = []
            for i in range(len(boxes)):
                targets.append({'boxes': boxes[i].to(device), 'labels': labels[i].to(device)})
            #targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            #Forward propagation
            try:
                out = model(images.to(device), targets)
                
                losses = sum(loss for loss in out.values())

                #Reseting Gradients
                optimizer.zero_grad()
                
                #Back propagation
                losses.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                
                #Average loss
                loss_value = losses.item()
                total_epoch_loss += loss_value
            except ValueError:
                print("Skipped because batch with no BBs")

        test_loss = test()

        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), os.path.join(SAVED_MODEL_DIR, SAVED_MODEL_NAME))
            new_best = f" - New Best Model"

        lr_scheduler.step(test_loss)    

        lr = round(optimizer.param_groups[0]['lr'], 10)

        epoch_train_loss = total_epoch_loss / len(train_data_loader)
        train_loss_logger.append(epoch_train_loss)
        test_loss_logger.append(test_loss)
        time_elapsed = time.time() - start_time

        print(f'Epoch [{epoch+1}/{epochs}] - lr: {lr} - Train Loss: {epoch_train_loss:.4f} - Test Loss: {test_loss:.4f} - Time: {time_elapsed:.2f}s{new_best}')

        if lr == LEARNING_RATE / 1e3:
            print("Training Finished Early on Plateau")
            break


def visualise(num_images):
    count = 0
    break_outer = False

    # Check it is working with a single sample image
    for images, _, _ in test_data_loader:
        for image in images:
            image = torch.unsqueeze(image, dim=0)
            model.eval()

            out = model(image.to(device))

            boxes = out[0]['boxes'].cpu().detach()
            labels = out[0]['labels'].cpu().detach()
            scores = out[0]['scores'].cpu().detach()

            nms_boxes_indices, nms_count = nms(boxes=boxes, scores=scores)
            
            image = image[0]
            image = np.moveaxis(image.numpy(),0,2)
            image = 255 * image
            image = image.astype(np.uint8)
            image = T.ToPILImage()(image)
            img1 = ImageDraw.Draw(image)

            for i in range(len(nms_boxes_indices)):
                index = nms_boxes_indices[i]
                box = boxes[index].numpy()
                label = labels[index].numpy()
                score = scores[index].numpy()

                buffer = -11 if box[1] <= 10 else 11
                img1.rectangle(box, outline ="red")
                img1.text([box[0], box[1] - buffer], face_classes[label])

            image.show() 
            count += 1  

            if count == num_images:
                break_outer = True
                break

        if break_outer:
            break


if __name__ == "__main__":

    # Faster - RCNN Model - pretrained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = len(face_classes)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    #Retriving all trainable parameters from model (for optimizer)
    params = [p for p in model.parameters() if p.requires_grad]
    #optimizer = torch.optim.SGD(params, lr = 5e-3, momentum = 0.9)
    optimizer = torch.optim.Adam(params, lr = LEARNING_RATE)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3)

    model.to(device)

    if TRAIN:
        train(EPOCHS)

    # Load best network and test
    model.load_state_dict(torch.load(os.path.join(SAVED_MODEL_DIR, SAVED_MODEL_NAME)))
    
    visualise(5)


