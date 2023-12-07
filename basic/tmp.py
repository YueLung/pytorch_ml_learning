import torch

# x = torch.randn(4, 4)
# z = x.view(2, -1)  # 确定一个维度，-1的维度会被自动计算
# print(x.item())


# print(x)
# print(x[:, 1])

# y = x.view(16)
# # print(y)
# print(x.size(), y.size())

# import numpy as np

# a = np.ones(5)
# b = torch.from_numpy(a)

# np.add(a, 1, out=a)
# print(a)
# print(b)

x = torch.ones(2, 2, requires_grad=True)
# print(x)

y = x + 2
# print(y)
# print(y.grad_fn)  # y就多了一个AddBackward

z = y * y * 3
out = z.mean()

print(z)  # z多了MulBackward
print(out)  # out多了MeanBackward

out.backward()
print(x.grad)  # out=0.25*Σ3(x+2)^2
