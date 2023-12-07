import torch
import torchvision
from torch.utils.data import DataLoader

from module.mnist.model import MNISTModel


mnistModel = MNISTModel()
mnistModel.load_state_dict(torch.load("./project/module/mnist/model.ckpt"))
mnistModel.eval()

test_data = torchvision.datasets.MNIST(
    root="./project/module/mnist/data",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=False,
)
test_loader = DataLoader(test_data, batch_size=5000, shuffle=False)

# test_inputs = test_data.data[:1000].type(torch.FloatTensor)
# test_labels = test_data.targets[:1000]

correct = 0
for data, taget in test_loader:
    test_output = mnistModel(data)

    pred_y = torch.max(test_output, 1)[1].data
    batch_correct = (pred_y == taget).sum().item()
    correct += batch_correct
    print(batch_correct)

print(correct / len(test_loader.dataset) * 100)
# print(test_output)
# print(torch.max(test_output, 1)[0])


# print(pred_y.numpy(), "prediction number")
# print(test_labels.numpy(), "real number")

# 比較兩個張量的元素是否相等
# correct = torch.eq(pred_y, test_labels)

# 計算準確率
# accuracy = torch.mean(correct.float()).item() * 100
# print(f"accuracy = {accuracy}")

# with torch.no_grad():
#     for inputs, labels in test_loader:
#         pred = mnistModel(inputs)
