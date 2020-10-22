import torch
import torchvision
import torch.nn as nn
import torch.functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import time

from data.data_loader import train_data_loader, test_data_loader, face_classes

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


def train():
    epochs = 10
    train_loss_logger = []
    test_loss_logger = []

    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        total_epoch_loss = 0
        
        #Retriving Mini-batch
        for images, boxes, labels in train_data_loader:

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

        #test_loss = test()
        test_loss = 0
        
        #lr_scheduler.step()    
        
        epoch_train_loss = total_epoch_loss / len(train_data_loader)
        train_loss_logger.append(epoch_train_loss)
        test_loss_logger.append(test_loss)
        time_elapsed = time.time() - start_time

        print(f'Epoch [{epoch+1}/{epochs}] - Train Loss: {epoch_train_loss:.4f} - Test Loss: {test_loss:.4f} - Time: {time_elapsed:.2f}s')
                
        # torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'loss': epoch_train_loss
        #         }, "checkpoint.pth")

if __name__ == "__main__":

    # Faster - RCNN Model - pretrained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = len(face_classes)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    #Retriving all trainable parameters from model (for optimizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(params, lr = 0.005, momentum = 0.9)
    model.to(device)

    train()

    test_loss = test()
    print(test_loss)


