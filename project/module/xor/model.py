import torch.nn.functional as F
import torch.nn as nn


class XorModel(nn.Module):
    def __init__(self):
        super(XorModel, self).__init__()
        self.layer1 = nn.Linear(
            in_features=2,
            out_features=2,
        )
        self.sigmoid = nn.Sigmoid()
        self.layer2 = nn.Linear(
            in_features=2,
            out_features=1,
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.sigmoid(x)
        x = self.layer2(x)
        return x
