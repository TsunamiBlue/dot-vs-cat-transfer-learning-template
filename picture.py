import cv2
import os
import torchvision.transforms as T
from torch.utils.data import Dataset
from utils.utils import resize_img, enhance
from PIL import Image

class config():
    perspective = 0.1
    IMAGE_SHAPE = (240,320)
    scale = 0.3
    rot = 30

transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5,0.5,0.5],[1/0.225,1/0.225,1/0.225])
        ])


class Picture(Dataset):
    def __init__(self, root, transforms=None, train=True):
        self.train = train
        images = os.listdir(root)
        self.images = [os.path.join(root, image) for image in images if image.endswith('.jpg') or image.endswith('.png')]
        self.transforms = transforms

    def __getitem__(self, index):
        image_path = self.images[index]
        # print(image_path)
        cv_image = cv2.imread(image_path)
        # print(cv_image.shape,image_path)
        # cv_image = Image.fromarray(cv2.cvtColor(cv_image,cv2.COLOR_BGR2RGB))
        re_img = resize_img(cv_image, config.IMAGE_SHAPE)
        # re_img = cv2.resize(cv_image,(config.IMAGE_SHAPE[1],
        # config.IMAGE_SHAPE[0]))
        # re_img = transform_handle(cv_image)
        # re_img = cv2.cvtColor(np.asarray(re_img),cv2.COLOR_RGB2BGR)
        tran_img, tran_mat = enhance(re_img, config)

        # cv2.imwrite('re_+' + str(index) +'.jpg',re_img)
        # cv2.imwrite('tran_'+ str(index) +'.jpg',tran_img)
        if self.transforms:
            re_img = Image.fromarray(cv2.cvtColor(re_img, cv2.COLOR_BGR2RGB))
            source_img = self.transforms(re_img)
            tran_img = Image.fromarray(cv2.cvtColor(tran_img, cv2.COLOR_BGR2RGB))
            des_img = self.transforms(tran_img)
        # else:
        #     re_img = Image.fromarray(cv2.cvtColor(re_img,cv2.COLOR_BGR2RGB))
        #     tran_img = Image.fromarray(cv2.cvtColor(tran_img,cv2.COLOR_BGR2RGB))
        #     image_array1 = np.asarray(re_img)
        #     image_array2 = np.asarray(tran_img)
        #     source_img = torch.from_numpy(image_array1)
        #     des_img = torch.from_numpy(image_array2)
        # if self.train:

        return source_img, des_img, tran_mat
        # else:
        # return source_img,des_img,tran_mat

    def __len__(self):
        return len(self.images)
