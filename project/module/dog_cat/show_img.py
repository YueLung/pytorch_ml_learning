import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt


Batch_Size = 50
Num_Workers = 0

Path_Train = "./project/module/dog_cat/Cat_Dog_data/test"

train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

train_data = datasets.ImageFolder(Path_Train, transform=train_transforms)

print(train_data.class_to_idx)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=Batch_Size, num_workers=Num_Workers, shuffle=True
)

images, labels = next(iter(train_loader))

print(images.size())
print(labels.size())


classes = ["cat", "dog"]
mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])


def denormalize(image):
    # image = transforms.Normalize(-mean / std, 1 / std)(image)  # denormalize
    image = image.permute(1, 2, 0)  # Changing from 3x224x224 to 224x224x3
    image = torch.clamp(image, 0, 1)
    return image


# helper function to un-normalize and display an image
def imshow(img):
    img = denormalize(img)
    plt.imshow(img)


# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = next(dataiter)
# convert images to numpy for display

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 8))

# display 20 images
for idx in range(20):
    ax = fig.add_subplot(2, 10, idx + 1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title("{} ".format(classes[labels[idx]]))
plt.show()
