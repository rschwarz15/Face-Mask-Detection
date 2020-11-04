import os
import json
import PIL.Image as Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader 
import torchvision.transforms as T
import albumentations as A
import matplotlib.pyplot as plt

BATCH_SIZE = 4
NUM_WORKERS = 4
RESIZE = 300
CROP = 256

classes = ['background', 'face_with_mask', 'mask_colorful',
           'face_no_mask', 'face_with_mask_incorrect', 'mask_surgical',
           'face_other_covering', 'scarf_bandana', 'eyeglasses',
           'helmet', 'face_shield', 'sunglasses',
           'hood', 'hat', 'goggles', 'hair_net', 'hijab_niqab',
           'other', 'gas_mask', 'balaclava_ski_mask', 'turban']

face_classes = ['background', 'face_with_mask', 'face_no_mask',
                'face_with_mask_incorrect', 'face_other_covering']


class FaceMaskDetectionDataset(Dataset):

    def __init__(self, training=True, test_size=100, transforms=None, class_list=face_classes, all_faces_one_class=True, only_images_with_mask_incorrect=False):
        super().__init__()

        self.training = training
        self.test_size = test_size
        self.unlabeled_data_size = 1_698
        self.transforms = transforms
        self.class_list = class_list
        self.all_faces_one_class = all_faces_one_class
        self.only_images_with_mask_incorrect = only_images_with_mask_incorrect

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
        image = np.array(image) / 255.0

        # Get Annotations
        annotation_path = os.path.join(self.annotations_dir, self.image_names[idx] + ".json")
        boxes = []
        labels = []

        with open(annotation_path) as f:
            annotations = json.load(f)

            for ann in annotations['Annotations']:
                box = ann['BoundingBox']
                classname = ann['classname']

                # prevent zero area boxes
                skip = False
                if box[0] == box[2] or box[1] == box[3]:
                    skip = True

                # if specified just include face classes
                if classname not in self.class_list:
                    skip = True

                if not skip:
                    boxes.append(box)

                    if self.all_faces_one_class:
                        labels.append(1) # map all face classes to the same class
                    else:
                        labels.append(face_classes.index(classname))

        # Return none for images if only including those that have face_with_mask_incorrect
        # collate_fn deals with the None's
        #
        # This occurs before cropping so it is possible that the
        # face_with_mask_incorrect box gets removed during data loading
        if self.only_images_with_mask_incorrect:
            face_with_mask_incorrect_idx = 3

            if face_with_mask_incorrect_idx not in labels:
                return None, None, None

        if self.transforms:
            transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
            image = transformed['image']
            boxes = list(transformed['bboxes'])
            labels = transformed['labels']

        # If no boxes, create a box that is the entire image and type background
        if len(boxes) == 0:
            boxes.append((0, 0, 1, 1))
            labels.append(0)

        image = torch.tensor(image, dtype=torch.float).permute(2, 0, 1).contiguous()
        boxes = torch.tensor(boxes, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.int64)

        return image, boxes, labels


def collate_fn(batch):
    # From
    # https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/43fd8be9e82b351619a467373d211ee5bf73cef8/datasets.py#L60
    """
    Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
    This describes how to combine these tensors of different sizes. We use lists.
    :param batch: an iterable of N sets from __getitem__()
    :return: a tensor of images, lists of varying-size tensors of bounding boxes and labels
    """

    images = list()
    boxes = list()
    labels = list()

    for b in batch:
        if b[0] is not None:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

    if len(images) > 0:
        images = torch.stack(images, dim=0)

    return images, boxes, labels


train_transform = A.Compose([
    A.Resize(width=RESIZE, height=RESIZE),
    # A.RandomCrop(width=CROP, height=CROP),
    # A.HorizontalFlip(p=0.5),
    # A.ShiftScaleRotate(rotate_limit=10)
],
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

test_transform = A.Compose([
    A.Resize(width=RESIZE, height=RESIZE),
    # A.CenterCrop(width=CROP, height=CROP),
],
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

train_dataset = FaceMaskDetectionDataset(training=True, transforms=train_transform)
test_dataset = FaceMaskDetectionDataset(training=False, transforms=test_transform)
#train_dataset_only_mask_incorrect = FaceMaskDetectionDataset(training=True, transforms=train_transform, only_images_with_mask_incorrect=True)

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

# train_data_loader_only_mask_incorrect = DataLoader(
#     train_dataset_only_mask_incorrect,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     num_workers=NUM_WORKERS,
#     collate_fn=collate_fn)

# Check counts of each class
if __name__ == "__main__":
    #counts = [0] * len(face_classes)
    #label_list = face_classes

    counts = [0] * 2
    label_list = ["Background", "Faces"]

    loader = train_data_loader
    for _, _, batch_labels in loader:
        for labels in batch_labels:
            for label in labels:
                counts[label] += 1

    plt.ylabel('Count')
    plt.bar(label_list, counts)
    plt.show()
    
    print(face_classes)
    print(counts)

