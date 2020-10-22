from data.data_loader import train_data_loader, test_data_loader, face_classes
from PIL import Image, ImageDraw
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    for batch_image, batch_boxes, batch_labels in train_data_loader:
        for i in range(len(batch_image)):
            image = batch_image[i]
            boxes = batch_boxes[i]
            labels = batch_labels[i]

            c, width, height = image.shape
            
            image = np.moveaxis(image.numpy(),0,2)
            image = 255 * image
            image = image.astype(np.uint8)
            image = T.ToPILImage()(image)

            img1 = ImageDraw.Draw(image)

            for j in range(len(boxes)):
                box = boxes[j]
                label = labels[j]

                print(box)
                print(label)

                img1.rectangle(box, outline ="red")
                img1.text([box[0], box[1] - 11], face_classes[label])

            image.show()

            break
        break