from pathlib import Path
import sys

from model import DogCatModel

sys.path.append("./project/core")

from device import get_best_device

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets

device = get_best_device()

Batch_Size = 50
Num_Workers = 0

Path_Train = "./project/module/dog_cat/Cat_Dog_data/test"
Path_Test = "./project/module/dog_cat/Cat_Dog_data/train"


train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

train_data = datasets.ImageFolder(Path_Train, transform=train_transforms)
test_data = datasets.ImageFolder(Path_Test)

print(train_data.class_to_idx)
print(test_data.class_to_idx)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=Batch_Size, num_workers=Num_Workers, shuffle=True
)

images, labels = next(iter(train_loader))

dogcatModel = DogCatModel()

torch.save(dogcatModel.state_dict(), "./project/module/dog_cat/model.ckpt")
