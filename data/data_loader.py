import os
import json
import PIL.Image as Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class FaceMaskDetectionDataset(Dataset):

    def __init__(self, training = True, test_size = 100):
        super().__init__()

        self.training = training
        self.test_size = test_size
        self.unlabeled_data_size = 1_698

        self.images_dir = "data/images/"
        self.annotations_dir = "data/annotations/"

        if training:
            self.image_names = os.listdir(self.images_dir)[self.unlabeled_data_size + test_size:]
        else:
            self.image_names = os.listdir(self.images_dir)[self.unlabeled_data_size: self.unlabeled_data_size + test_size]


    def __len__(self):
        return len(self.images_dir)
        
    def __getitem__(self, idx: int):
        # Get Image
        image_path = os.path.join(self.images_dir, self.image_names[idx])
        image = Image.open(image_path) 

        # Get Associated Labels
        annotation_path = os.path.join(self.annotations_dir, self.image_names[idx] + ".json")

        with open(annotation_path) as f:
            annotations = json.load(f)
            # Filter for what we want
 
        return image, annotations
