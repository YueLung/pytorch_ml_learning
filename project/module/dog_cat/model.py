import torch.nn as nn
import torch.nn.functional as F


class DogCatModel(nn.Module):
    def __init__(self):
        super(DogCatModel, self).__init__()
        # self.flatten = nn.Flatten()
        self.cnn1 = nn.Conv2d(1, 10, kernel_size=5)  # (10,24,24) 28-4
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)  # (10,12,12)

        self.cnn2 = nn.Conv2d(10, 20, kernel_size=5)  # (20,8,8)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)  # (20,4,4)

        self.fc1 = nn.Linear(20 * 4 * 4, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.cnn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = x.view(-1, 320)

        x = self.fc1(x)
        x = self.fc2(x)
        return x
