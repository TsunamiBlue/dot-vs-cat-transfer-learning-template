import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv2d(3, 32, 3,stride=1,padding=1)
        # self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm2d(32)
        # self.relu = nn.ReLU(inplace=True)
        # self.pool1 = nn.MaxPool2d(2, 2)
        # self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(64)
        # self.fc1 = nn.Linear(56*56*64,1024)
        # self.bn3 = nn.BatchNorm1d(1024)
        # self.fc2 = nn.Linear(1024,1)


        self.network2d = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )

        self.network1d = nn.Sequential(
            nn.Linear(56*56*64,1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1024,2)
        )

    def forward(self, x):
        x = self.network2d(x)
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.pool1(x)

        # x = self.conv3(x)
        # x = self.bn2(x)
        # x = self.relu(x)
        # x = self.pool1(x)
        x = x.view(-1,56*56*64)
        x = self.network1d(x)
        return x


