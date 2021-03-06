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
import matplotlib.pyplot as plt
from tqdm import tqdm

from data.face_detection_data_loader import train_data_loader, test_data_loader, face_classes

SAVED_MODEL_DIR = "saved_models"
SAVED_MODEL_FINAL_NAME = "faceDetectorFinal.pt"
SAVED_MODEL_BEST_NAME = "faceDetectorBest.pt"
SAVE_OUTPUTS_DIR = "results/model_outputs/"
EPOCHS = 3         
OPTIMIZER = "SGD"       # SGD   or ADAM
SCHEDULER = "StepLR"    # Plateau or StepLR
StepLR_SIZE = 5         # StepLR step size
LEARNING_RATE = 5e-3    # SGD 5e-3 ADAM 1e-4
TRAIN = True
VISUALISE = True
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

    epoch_test_loss = total_epoch_loss / len(test_data_loader)

    return epoch_test_loss


def train(epochs, dataloader, train_loss_logger, test_loss_logger):
    best_test_loss = 999
    training_start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_epoch_loss = 0
        new_best = ""
        
        #Retriving Mini-batch
        for images, boxes, labels in tqdm(dataloader, dynamic_ncols=True):

            # To accomodate for train_dataset_only_mask_incorrect skip those with empty images 
            if len(images) == 0:
                continue

            #Loading images & targets on device
            targets = []
            for i in range(len(boxes)):
                targets.append({'boxes': boxes[i].to(device), 'labels': labels[i].to(device)})
            #targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            #Forward propagation
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

        test_loss = test()

        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), os.path.join(SAVED_MODEL_DIR, SAVED_MODEL_BEST_NAME))
            new_best = f" - New Best Model"

        if SCHEDULER == "Plateau":
            lr_scheduler.step(total_epoch_loss) 
        else:
            lr_scheduler.step()   

        lr = round(optimizer.param_groups[0]['lr'], 10)

        epoch_train_loss = total_epoch_loss / len(train_data_loader)
        train_loss_logger.append(epoch_train_loss)
        test_loss_logger.append(test_loss)

        print(f'Epoch [{epoch+1}/{epochs}] - lr: {lr} - Train Loss: {epoch_train_loss:.4f} - Test Loss: {test_loss:.4f}{new_best}')

        if lr == LEARNING_RATE / 1e3:
            print("Training Finished Early on Plateau")
            break

    # Save the final model as well
    torch.save(model.state_dict(), os.path.join(SAVED_MODEL_DIR, SAVED_MODEL_FINAL_NAME))
    full_training_time = time.time() - training_start_time
    print(f'Training took {int(full_training_time//60)} minutes and {full_training_time - (full_training_time//60)*60:.2f} seconds')

    # Return training loss and testing loss for plotting
    return train_loss_logger, test_loss_logger


def plot_loss(train_loss_array, test_loss_array):
    min_train_loss_idx = np.argmin(train_loss_array)
    min_train_loss = train_loss_array[min_train_loss_idx]
    print(f"Minimum training loss - index: {min_train_loss_idx}, value : {min_train_loss:.3f}")

    min_test_loss_idx = np.argmin(test_loss_array)
    min_test_loss = test_loss_array[min_test_loss_idx]
    print(f"Minimum test loss - index: {min_test_loss_idx}, value : {min_test_loss:.3f}")

    print("Generating results")

    plt.figure(figsize=(10, 10))

    plt.plot(train_loss_array, label='Training')
    plt.scatter(min_train_loss_idx, min_train_loss, s=20, c='blue', marker='d')

    plt.plot(test_loss_array, label='Validation')
    plt.scatter(min_test_loss_idx, min_test_loss, s=20, c='red', marker='d')

    plt.title(
        f"Training and Validation Loss\nEpochs: {EPOCHS} lr: {LEARNING_RATE}")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def visualise(num_images):
    count = 0
    break_outer = False
    model.eval()

    # Check it is working with a single sample image
    for images, _, _ in test_data_loader:
        for image in images:
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

                buffer = -11 if box[1] <= 10 else 0
                img1.rectangle(box, outline ="red")
                img1.text([box[0], box[1] - buffer], face_classes[label])

            #image.show() 
            image.save(SAVE_OUTPUTS_DIR + f"test{count}.jpg")
            count += 1  

            if count == num_images:
                break_outer = True
                break

        if break_outer:
            break


if __name__ == "__main__":
    # Faster RCNN Model - pretrained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2 # bg and face

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    #Retriving all trainable parameters from model (for optimizer)
    params = [p for p in model.parameters() if p.requires_grad]

    # Optimizer selection
    if OPTIMIZER == "SGD":
        optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9)
    elif OPTIMIZER == "ADAM":
        optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
    else:
        raise ValueError("Please select a valid optimizer")
    
    if SCHEDULER == "Plateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    elif SCHEDULER == "StepLR":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=StepLR_SIZE)
    else:
        raise ValueError("Please select a valid scheduler")
    model.to(device)

    if TRAIN:
        train_loss_logger, test_loss_logger = [], []

        # train on all data
        train_loss_logger, test_loss_logger = train(
            EPOCHS, 
            train_data_loader, 
            train_loss_logger, 
            test_loss_logger)

        plot_loss(train_loss_logger, test_loss_logger)

    if VISUALISE:
        # Load best network and visualise
        model.load_state_dict(torch.load(os.path.join(SAVED_MODEL_DIR, SAVED_MODEL_BEST_NAME)))
        visualise(20)


