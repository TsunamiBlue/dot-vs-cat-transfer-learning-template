import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F




# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# resnet18 = models.resnet18(pretrained=True)
# fc_features = resnet18.fc.in_features
# resnet18.fc = nn.Linear(fc_features, 2)



class Pretrained_ResNet18(nn.Module):
    def __init__(self):
        super().__init__()


        self.pretrained_ResNet18 = models.resnet18(pretrained=True)
        fc_features = self.pretrained_ResNet18.fc.in_features

        self.finalFCN = nn.Linear(fc_features, 2)




    def forward(self, x):
        x = self.pretrained_ResNet18(x)

        x = self.finalFCN(x)
        return x


