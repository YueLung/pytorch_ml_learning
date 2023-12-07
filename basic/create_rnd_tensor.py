import torch

# 隨機

# 隨機創造維度為 (2, 3) 的張量，數值為無法控制範圍的浮點
print(torch.empty((2, 3)))
print()

# 隨機創造維度為 (2, 3) 的張量，數值為介於 0 到 1 之間的浮點
print(torch.rand(2, 3))
print()

# 隨機創造維度為 (2, 3) 的張量，數值為介於 0 到 10 之間的浮點
print(torch.rand(2, 3) * 10)
print()

# 隨機創造維度為 (2, 3) 的張量，數值為介於 -5 到 5 之間的浮點數
print(torch.rand(2, 3) * 10 - 5)
print()

# 隨機創造維度為 (2, 3) 的張量，分佈為平均值為 0 標準差為 1 的常態分佈
print(torch.randn(2, 3))
print()

# 隨機創造維度為 (2, 3) 的張量，數值為介於 -5 到 5 之間的浮點數
print(torch.randint(-5, 5, size=(2, 3)))
