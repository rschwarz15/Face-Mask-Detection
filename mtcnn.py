import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from data.data_loader import train_data_loader, test_data_loader, face_classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mtcnn = MTCNN(keep_all=True, device=device)
 
if __name__ == "__main__":   
    # Check it is working with a single sample image
    count = 0
    total_tests = 1
    break_outer = False

    for batch_image, _, _ in test_data_loader:
        for i in range(len(batch_image)):
            image = batch_image[i].permute(1,2,0)
            #plt.imshow(image)
            #plt.show()

            boxes, _ = mtcnn.detect(image)
            print(boxes)

            count += 1

            if count == total_tests:
                break_outer = True
                break
        if break_outer:
            break

        
