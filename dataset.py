import cv2
import os
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image





class Picture(Dataset):
    def __init__(self, root, transforms=None):
        images = os.listdir(root)
        self.images = [os.path.join(root, image) for image in images]
        self.transforms = transforms

    def __getitem__(self, index):
        image_path = self.images[index]

        cv_image = cv2.imread(image_path)

        if self.transforms:
            source_img = self.transforms(cv_image)
        else:
            source_img = cv_image


        tmp = image_path.split("\\")
        filename = tmp[-1]
        tmp = tmp[-1].split(".")
        label_str = tmp[0]

        if label_str == "dog":
            label = 1
        elif label_str == "cat":
            label = 0
        else:
            label = -1

        # print(filename,label)

        return source_img,filename,label

    def __len__(self):
        return len(self.images)
