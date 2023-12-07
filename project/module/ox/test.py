# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# from torchvision import transforms, datasets

# # 数据预处理和加载
# transform = transforms.Compose([
#     transforms.Resize((64, 64)),  # 调整图像大小
#     transforms.ToTensor(),        # 转换为张量
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
# ])

# train_data = datasets.ImageFolder('data/train', transform=transform)
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

# # 构建CNN模型
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, 3)
#         self.fc1 = nn.Linear(64 * 62 * 62, 2)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         return x

# model = Net()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001)

# # 训练模型
# for epoch in range(10):
#     for i, data in enumerate(train_loader, 0):
#         inputs, labels = data
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

# # 用模型进行预测
# test_data = datasets.ImageFolder('data/test', transform=transform)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False)

# model.eval()
# with torch.no_grad():
#     for data in test_loader:
#         images, labels = data
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)

#         for prediction in predicted:
#               if prediction.item() == 0:
#                   print("Predicted: o")
#               else:
#                   print("Predicted: x")


import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preprocessing and loading
transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

train_data = datasets.ImageFolder("./project/module/ox/data/train", transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

# Build the CNN model and move it to the GPU


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.fc1 = nn.Linear(64 * 62 * 62, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


model = Net().to(device)  # Move the model to the GPU

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Training the model on the GPU
for epoch in range(1000):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to the GPU
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Using the model for inference on the GPU
test_data = datasets.ImageFolder("./project/module/ox/data/test", transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False)

model.eval()
with torch.no_grad():
    for data in test_loader:
        (
            images,
            labels,
        ) = data

        images, labels = images.to(device), labels.to(device)  # Move data to the GPU
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        for prediction in predicted:
            if prediction.item() == 0:
                print(f"File: , Predicted: o")
            else:
                print(f"File: , Predicted: x")
