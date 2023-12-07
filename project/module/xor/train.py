import time
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from model import XorModel
from dataset import XorDataset

xorDataset = XorDataset()
xor_dataloader = DataLoader(xorDataset, batch_size=2, shuffle=True)
# for inputs, labels in xor_dataloader:
#     print(inputs, labels)

xorModel = XorModel()

criterion = nn.MSELoss()
optimizer = SGD(
    xorModel.parameters(),
    lr=0.02,
)

start_time = time.time()
losses = []

n_epoch = 20000
for epoch in range(n_epoch):
    total_loss = 0
    for inputs, labels in xor_dataloader:
        # print(f"inputs: {inputs}  labels: {labels}")
        # print(inputs.size())
        pred = xorModel(inputs)
        loss = criterion(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(f"{loss.item()}   {float(loss)}")
        total_loss += float(loss)

    losses.append(total_loss)
    if epoch % 1000 == 0:
        print(f"epoch = {epoch} loss = {total_loss}")

end_time = time.time()
training_time = end_time - start_time
print(f"Training time: {training_time} seconds")

# 绘制损失图表
plt.plot(losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.legend()
plt.show()


# 获取模型的权重
model_weights = xorModel.state_dict()

# 打印权重信息
# for key, value in model_weights.items():
#     print(f"{key}: {value}")

torch.save(xorModel.state_dict(), "./project/module/xor/model.ckpt")
