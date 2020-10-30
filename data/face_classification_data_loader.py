import os
import json
from PIL import ImageDraw
import PIL.Image as Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader 
import torchvision.transforms as T
import matplotlib.pyplot as plt

BATCH_SIZE = 48
NUM_WORKERS = 4
IMG_SIZE = 128

face_classes = ['face_with_mask', 'face_no_mask',
                'face_with_mask_incorrect', 'face_other_covering']

class FaceMaskDetectionDataset(Dataset):

    def __init__(self, training=True, test_size=100, transforms=None,):
        super().__init__()

        self.training = training
        self.test_size = test_size
        self.unlabeled_data_size = 1_698
        self.transforms = transforms

        self.images_dir = "data/images/"
        self.annotations_dir = "data/annotations/"

        if training:
            self.image_names = os.listdir(self.images_dir)[self.unlabeled_data_size + test_size:]
        else:
            self.image_names = os.listdir(self.images_dir)[self.unlabeled_data_size: self.unlabeled_data_size + test_size]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx: int):
        # Get Image
        image_path = os.path.join(self.images_dir, self.image_names[idx])
        image = Image.open(image_path)
        image = np.array(image)

        # Get Annotations
        annotation_path = os.path.join(self.annotations_dir, self.image_names[idx] + ".json")
        image_crops = []
        labels = []

        with open(annotation_path) as f:
            annotations = json.load(f)

            # For each annotation that is a face crop the image and return that
            for ann in annotations['Annotations']:
                box = ann['BoundingBox']
                x1 = box[0]
                y1 = box[1]
                x2 = box[2]
                y2 = box[3]
                classname = ann['classname']
                
                # prevent zero area boxes
                skip = False
                if box[0] == box[2] or box[1] == box[3]:
                    skip = True

                # if specified just include face classes
                if classname not in face_classes:
                    skip = True

                if not skip:
                    labels.append(face_classes.index(classname))

                    image_crop = image[y1:y2+1, x1:x2+1, :] # W, H, C
                    image_crops.append(image_crop)

        if self.transforms:
            for i in range(len(image_crops)):
                image_crops[i] = self.transforms(image_crops[i])

        labels = torch.tensor(labels, dtype=torch.int64)

        return image_crops, labels


def collate_fn(batch):
    # From
    # https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/43fd8be9e82b351619a467373d211ee5bf73cef8/datasets.py#L60
    """
    Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader)
    This describes how to combine these tensors of different sizes. We use lists.
    :param batch: an iterable of N sets from __getitem__()
    :return: a tensor of images and labels
    """

    image_crops = list()
    labels = list()

    for b in range(len(batch)):
        for i in range(len(batch[b][0])):
            image_crops.append(batch[b][0][i])
            labels.append(batch[b][1][i])

    # print(np.array(image_crops[0]).shape)
    # print(np.array(image_crops[1]).shape)
    
    if len(image_crops) > 0:
        image_crops = torch.stack(image_crops, dim=0)
        labels = torch.stack(labels, dim=0)

    return image_crops, labels


train_transform = T.Compose([
    T.ToPILImage(),
    T.RandomHorizontalFlip(p=0.5),
    T.Resize((128, 128)),
    T.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    T.ToTensor()
])

test_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((128, 128)),
    T.ToTensor()
])

train_dataset = FaceMaskDetectionDataset(training=True, transforms=train_transform)
test_dataset = FaceMaskDetectionDataset(training=False, transforms=test_transform)

train_data_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn)

test_data_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn)

# Check counts of each class
if __name__ == "__main__":
    displayCounts = True
    displayImage = False

    loader = train_data_loader

    if displayCounts:
        counts = [0] * len(face_classes)

        for _, batch_labels in loader:
            for label in batch_labels:
                counts[label] += 1

        plt.ylabel('Count')
        plt.bar(face_classes, counts)
        plt.show()
        
        print(face_classes)
        print(counts)
    
    if displayImage:
        batch_image_crops, batch_labels = next(iter(train_data_loader))

        for i in range(len(batch_image_crops)):
            image_crop = batch_image_crops[i]          
            image_crop = np.moveaxis(image_crop.numpy(),0,2)
            image_crop = 255 * image_crop
            image_crop = image_crop.astype(np.uint8)
            image_crop = T.ToPILImage()(image_crop)
            
            plt.title(face_classes[batch_labels[i]])
            plt.imshow(image_crop)
            plt.show()



