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

BATCH_SIZE = 4
NUM_WORKERS = 4
RESIZE = 128
CROP = 96

classes = ['background', 'face_with_mask', 'mask_colorful',
                'face_no_mask', 'face_with_mask_incorrect', 'mask_surgical',
                'face_other_covering', 'scarf_bandana', 'eyeglasses',
                'helmet', 'face_shield', 'sunglasses',
                'hood', 'hat', 'goggles', 'hair_net', 'hijab_niqab',
                'other', 'gas_mask', 'balaclava_ski_mask', 'turban']

face_classes = ['background', 'face_with_mask', 'face_no_mask',
                'face_with_mask_incorrect', 'face_other_covering']

class FaceMaskDetectionDataset(Dataset):

    def __init__(self, training = True, test_size = 100, transforms=None):
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
        image = np.array(image) / 255.0

        # Get Annotations
        annotation_path = os.path.join(self.annotations_dir, self.image_names[idx] + ".json")
        boxes = []
        labels = []

        with open(annotation_path) as f:
            annotations = json.load(f)
            # Filter for what we want
            
            for ann in annotations['Annotations']:
                box = ann['BoundingBox']
                classname = ann['classname']

                # only include faces for now
                if classname in face_classes:
                    boxes.append(box)
                    labels.append(face_classes.index(classname))
        
        if self.transforms:
            transformed = self.transforms(image=image, bboxes=boxes, labels=labels)  
            image = transformed['image']
            boxes = list(transformed['bboxes'])
            labels = transformed['labels'] 

        image = torch.tensor(image, dtype=torch.float).permute(2,0,1).contiguous()
        boxes = torch.tensor(boxes, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.int64)

        return image, boxes, labels

    def collate_fn(self, batch):
        # From 
        # https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/43fd8be9e82b351619a467373d211ee5bf73cef8/datasets.py#L60
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)

        return images, boxes, labels

train_transform = A.Compose([
    A.Resize(width=RESIZE, height=RESIZE),
    A.RandomCrop(width=CROP, height=CROP),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(rotate_limit=15)
    ], 
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
)

test_transform = A.Compose([
    A.Resize(width=RESIZE, height=RESIZE),
    A.CenterCrop(width=CROP, height=CROP),
    ], 
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
)

train_dataset = FaceMaskDetectionDataset(training=True, transforms=train_transform)
test_dataset = FaceMaskDetectionDataset(training=False, transforms=test_transform)

train_data_loader = DataLoader(
    train_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True,
    num_workers = NUM_WORKERS,
    collate_fn=train_dataset.collate_fn
)

test_data_loader = DataLoader(
    test_dataset,
    batch_size = BATCH_SIZE,
    shuffle = False,
    num_workers = NUM_WORKERS,
    collate_fn=test_dataset.collate_fn
)