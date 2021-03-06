import os
from PIL import ImageDraw
import numpy as np
from numpy.core.fromnumeric import argmax
import torch
import torchvision
import torch.nn as nn
import torch.functional as F
import torchvision.transforms as T
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
plt.rcParams.update({'font.size': 10})

from data.face_classification_data_loader import train_data_loader, test_data_loader, face_classes

# SAVED_MODEL_DIR = "saved_models"
# SAVED_MODEL_FINAL_NAME = "faceClassificationFinal.pt"
# SAVED_MODEL_BEST_NAME = "faceClassificationBest.pt"
# SAVE_OUTPUTS_DIR = "results/model_outputs/"
EPOCHS = 50
OPTIMIZER = "SGD"      # SGD   or ADAM
SCHEDULER = "StepLR"    # Plateau or StepLR
StepLR_SIZE = 35       # StepLR step size
LEARNING_RATE = 1e-2    # SGD 5e-3 ADAM 1e-4
TRAIN = False
VISUALISE = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVED_MODEL_DIR = "saved_models"
MODEL_NAME = f"Face_Classification_{OPTIMIZER}_{SCHEDULER}_{EPOCHS}_{LEARNING_RATE}"
SAVED_MODEL_FINAL_NAME = MODEL_NAME + "_final.pt"
SAVED_MODEL_BEST_NAME = MODEL_NAME + "_best.pt"
SAVE_TRAINING_GRAPHS = "results/training_graphs"
SAVE_OUTPUTS_DIR = f"results/model_outputs/{MODEL_NAME}"

# Create saved model path if it does not exist
if not os.path.exists(SAVED_MODEL_DIR):
    os.makedirs(SAVED_MODEL_DIR)

# Create results path if it does not exist
if not os.path.exists(SAVE_OUTPUTS_DIR):
    os.makedirs(SAVE_OUTPUTS_DIR)

def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    return acc


def test():
    #initialise counters
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()                              #Set Network in test mode
    
    for i, (x, y) in enumerate(test_data_loader):
        
        x = x.to(device)                    # Load images to device
        y = y.long().to(device)             # Load labels to device

        with torch.no_grad():
            fx = model(x)                     # Forward Pass   
        loss = loss_fun(fx, y)              # Find Loss  
        acc = calculate_accuracy(fx, y)     # Find Accuracy
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(test_data_loader), epoch_acc / len(test_data_loader)


def train(epochs, train_loss_logger=[], test_loss_logger=[], train_acc_logger=[], test_acc_logger=[]):
    best_test_acc = 0
    training_start_time = time.time()

    for epoch in range(epochs):
        new_best = ""
        epoch_loss = 0
        epoch_acc = 0
        
        model.train()                             #Set Network in train mode
        
        for x, y in tqdm(train_data_loader, dynamic_ncols=True):
            
            x = x.to(device)                    # Load images to device
            y = y.to(device)                    # Load labels to device

            fx = model(x)                       # Forward Pass   
            loss = loss_fun(fx, y)              # Find Loss  
            acc = calculate_accuracy(fx, y)     # Find Accuracy
            
            model.zero_grad()                   # Zero Gradents
            loss.backward()                     # Backpropagate Gradents
            optimizer.step()                    # Do a single optimization step
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()

        epoch_loss /=  len(train_data_loader)
        epoch_acc /=  len(train_data_loader)

        test_loss, test_acc = test()

        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), os.path.join(SAVED_MODEL_DIR, SAVED_MODEL_BEST_NAME))
            new_best = f" - New Best Model"

        # Step Learning Rate
        if SCHEDULER == "Plateau":
            lr_scheduler.step(epoch_loss) 
        else:
            lr_scheduler.step()   

        lr = round(optimizer.param_groups[0]['lr'], 10)

        # Log Data
        train_loss_logger.append(epoch_loss)
        test_loss_logger.append(test_loss)
        train_acc_logger.append(epoch_acc)
        test_acc_logger.append(test_acc)

        print(f'Epoch [{epoch+1}/{epochs}] - lr: {lr} - Train Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc*100:.2f}% - Test Loss: {test_loss:.4f} - Test Acc: {test_acc*100:.2f}%{new_best}')

        if lr == LEARNING_RATE / 1e3:
            print("Training Finished Early on Plateau")
            break

    # Save the final model as well
    torch.save(model.state_dict(), os.path.join(SAVED_MODEL_DIR, SAVED_MODEL_FINAL_NAME))
    full_training_time = time.time() - training_start_time
    print(f'Training took {int(full_training_time//60)} minutes and {full_training_time - (full_training_time//60)*60:.2f} seconds')

    # Return training loss and testing loss for plotting
    return train_loss_logger, test_loss_logger, train_acc_logger, test_acc_logger


def plot_loss(train_loss_array, test_loss_array, save=False, visual=False):
    min_train_loss_idx = np.argmin(train_loss_array)
    min_train_loss = train_loss_array[min_train_loss_idx]
    print(f"Minimum training loss - index: {min_train_loss_idx}, value : {min_train_loss:.3f}")

    min_test_loss_idx = np.argmin(test_loss_array)
    min_test_loss = test_loss_array[min_test_loss_idx]
    print(f"Minimum test loss - index: {min_test_loss_idx}, value : {min_test_loss:.3f}")

    plt.figure(figsize=(10, 10))

    plt.plot(train_loss_array, label='Training')
    plt.scatter(min_train_loss_idx, min_train_loss, s=20, c='blue', marker='d')

    plt.plot(test_loss_array, label='Testing')
    plt.scatter(min_test_loss_idx, min_test_loss, s=20, c='red', marker='d')

    plt.title(
        f"Training and Testing Loss\n\n"
        f"Epochs: {EPOCHS}, Optimiser: {OPTIMIZER}, LR: {LEARNING_RATE}, LR Scheduler: {SCHEDULER}\n"
        f"Minimum training loss - index: {min_train_loss_idx}, value: {min_train_loss:.3f}\n"
        f"Minimum test loss - index: {min_test_loss_idx}, value: {min_test_loss:.3f}"
    )
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    if visual:
        plt.show()
    if save:
        plt.savefig(os.path.join(SAVE_TRAINING_GRAPHS, MODEL_NAME + "_Loss.png"))
        plt.close()


def plot_acc(train_acc_array, test_acc_array, save=False, visual=False):
    max_train_acc_idx = np.argmax(train_acc_array)
    max_train_acc = train_acc_array[max_train_acc_idx]
    print(f"Max training accuracy - index: {max_train_acc_idx}, value : {max_train_acc*100:.2f}%")

    max_test_acc_idx = np.argmax(test_acc_array)
    max_test_acc = test_acc_array[max_test_acc_idx]
    print(f"Max test accuracy - index: {max_test_acc_idx}, value : {max_test_acc*100:.2f}%")

    plt.figure(figsize=(10, 10))
    train_acc_array = np.array(train_acc_array)
    test_acc_array = np.array(test_acc_array)

    plt.plot(train_acc_array*100, label='Training')
    plt.scatter(max_train_acc_idx, max_train_acc*100, s=20, c='blue', marker='d')

    plt.plot(test_acc_array*100, label='Testing')
    plt.scatter(max_test_acc_idx, max_test_acc*100, s=20, c='red', marker='d')

    plt.title(
        f"Training and Testing Accuracy\n\n"
        f"Epochs: {EPOCHS}, Optimiser: {OPTIMIZER}, LR: {LEARNING_RATE}, LR Step: {SCHEDULER}\n"
        f"Max training accuracy - index: {max_train_acc_idx}, value: {max_train_acc*100:.2f}%\n"
        f"Max test accuracy - index: {max_test_acc_idx},value : {max_test_acc*100:.2f}%"
    )
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    if visual:
        plt.show()
    if save:
        plt.savefig(os.path.join(SAVE_TRAINING_GRAPHS, MODEL_NAME + "_Accuracy.png"))
        plt.close()


def visualise(num_images, save=False, visual=False):
    count = 0
    break_outer = False
    model.eval()

    # Check it is working with a single sample image
    for b in range(len(test_data_loader)):
        batch_images, batch_labels = next(iter(test_data_loader))
    
        for ctr, i in enumerate(range(len(batch_images))):
            image = batch_images[i]
            correct_label = batch_labels[i]

            image = torch.unsqueeze(image, dim=0)
            
            out = model(image.to(device))
            predicted_output = torch.argmax(out)

            image = image[0]
            image = np.moveaxis(image.numpy(),0,2)
            image = 255 * image
            image = image.astype(np.uint8)
            image = T.ToPILImage()(image)
            img1 = ImageDraw.Draw(image)

            # print(f"{ctr} Predicted output: {predicted_output}")
            # print(f"{ctr} Correct output: {correct_label}")
            plt.title(f"predicted: {face_classes[predicted_output]}\ncorrect: {face_classes[correct_label]}")
            plt.imshow(image)
            if visual:

                plt.show()
            if save:
                plt.savefig(os.path.join(SAVE_OUTPUTS_DIR, f"test_{count}.jpg"))
                plt.close()
            count += 1  

            if count == num_images:
                break_outer = True
                break

        if break_outer:
            break


if __name__ == "__main__":
    # Resnet 18 Classification model
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 4) # Overwrite clasification layer from ImageNet

    # Define loss function
    class_counts = torch.tensor([4080, 1543, 142, 1346])
    weights = torch.true_divide(class_counts, class_counts.sum()) 
    weights = torch.true_divide(1.0, weights).to(device)
    loss_fun = nn.CrossEntropyLoss(weight=weights)

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
        train_loss_logger, test_loss_logger, train_acc_logger, test_acc_logger = train(EPOCHS)

        plot_loss(train_loss_logger, test_loss_logger, visual=False, save=True)
        plot_acc(train_acc_logger, test_acc_logger, visual=False, save=True)

    if VISUALISE:
        # Load best network and visualise
        model.load_state_dict(torch.load(os.path.join(SAVED_MODEL_DIR, SAVED_MODEL_BEST_NAME)))
        print(f"Loading model: {os.path.join(SAVED_MODEL_DIR, SAVED_MODEL_BEST_NAME)}")
        visualise(30, visual=False, save=True)


