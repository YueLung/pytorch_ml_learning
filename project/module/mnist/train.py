import sys


sys.path.append("./project/core")
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
from torch.optim import SGD
from device import get_best_device
from model import MNISTModel

device = get_best_device()

EPOCH = 3
BATCH_SIZE = 50
LR = 0.01
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
    root="./project/module/mnist/data",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)
data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

mnistModel = MNISTModel()
# mnistModel.to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = SGD(
    mnistModel.parameters(),
    lr=LR,
)

start_time = time.time()

losses = []
losses_cnt = []

for epoch in range(EPOCH):
    for batch_index, (inputs, labels) in enumerate(data_loader):
        # print(inputs.size())
        # print(labels.size())
        # inputs = inputs.to(device)
        # labels = labels.to(device)
        pred = mnistModel(inputs)
        # print(pred.size())
        loss = loss_func(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 100 == 0:
            # print(f"epoch: {epoch} batch_index:{batch_index} loss: {loss.item()}")
            losses.append(loss.item())
            losses_cnt.append(epoch * len(data_loader.dataset) + batch_index * 50)

end_time = time.time()
training_time = end_time - start_time
print(f"Training time: {training_time} seconds")

plt.plot(losses_cnt, losses, label="loss")
plt.xlabel("count")
plt.ylabel("loss")
plt.legend()
plt.show()

# print(loss)
# print(train_data.train_data.size())
# print(train_data.train_labels.size())
# plt.ion()
# for i in range(11):
#     plt.imshow(train_data.train_data[i].numpy(), cmap="gray")
#     plt.title("%i" % train_data.train_labels[i])
#     plt.pause(0.5)
# plt.show()

# print(train_data.train_data[0])
# print(train_data.targets[0])
torch.save(mnistModel.state_dict(), "./project/module/mnist/model.ckpt")
