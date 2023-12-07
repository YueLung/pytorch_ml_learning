import matplotlib.pyplot as plt
import torchvision


DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
    root="./project/module/mnist/data",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

print(train_data.data.size())
# print(train_data.data[0])
print(train_data.targets.size())

for i in range(1):
    plt.imshow(train_data.data[i].numpy(), cmap="gray")
    plt.title("%i" % train_data.targets[i])
    plt.pause(0.5)
plt.show()
