from data.data_loader import FaceMaskDetectionDataset
from PIL import Image, ImageDraw

if __name__ == "__main__":
    data = FaceMaskDetectionDataset(training=True)
    
    image, annotation = data[654]
    width, height = image.size

    img1 = ImageDraw.Draw(image)
    for annotation in annotation['Annotations']:
        bb = annotation['BoundingBox']
        classname = annotation['classname']

        img1.rectangle(bb, outline ="red")
        img1.text([bb[0], bb[1] - 11], classname)

    image.show()
