import torch

# 張量宣告

# 宣告 Tensor 變數
t1 = torch.tensor([1, 2, 3])
# 輸出 Tensor
print(t1)
# 輸出 True
print(type(t1) == torch.Tensor)
# 輸出 torch.int64
print(t1.dtype)
print()

# 宣告 Tensor 變數
t2 = torch.tensor([1.0, 2.0, 3.0])
# 輸出 Tensor
print(t2)
# 輸出 True
print(type(t2) == torch.Tensor)
# 輸出 torch.float32
print(t2.dtype)
print()

# 各種 dtype
# 輸出 torch.int8
print(torch.tensor([1, 2], dtype=torch.int8).dtype)
x = torch.tensor([1, 2], dtype=torch.int8)
print(x)
try:
    x[0] = 128
except Exception as e:
    # int8 的範圍為 -128 ~ 127
    # RuntimeError: value cannot be converted to type int8_t without overflow
    print(e)
print()
