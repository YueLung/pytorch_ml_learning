import torch

# size 屬性

# 宣告 Tensor 變數
t5 = torch.tensor(
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
    ]
)

# 輸出 Tensor
print(t5)
# 輸出 t5.size (4, 3)
print(t5.size())
print()

# 重新更改 t5.size
print(t5.reshape(3, 4))
# 輸出更改後的維度 (3, 4)
print(t5.reshape(3, 4).size())
print()
# 重新更改 t5.size
print(t5.view(3, 4))
# 輸出更改後的維度 (3, 4)
print(t5.view(3, 4).size())
print()

# 重新更改 t5.size
print(t5.reshape(2, 6))
# 輸出更改後的維度 (2, 6)
print(t5.reshape(2, 6).size())
print()
# 重新更改 t5.size
print(t5.view(2, 6))
# 輸出更改後的維度 (2, 6)
print(t5.view(2, 6).size())
print()

# 重新更改 t5.size
print(t5.reshape(2, 3, 2))
# 輸出更改後的維度 (2, 3, 2)
print(t5.reshape(2, 3, 2).size())
print()
# 重新更改 t5.size
print(t5.view(2, 3, 2))
# 輸出更改後的維度 (2, 3, 2)
print(t5.view(2, 3, 2).size())
# 自動計算第一個維度
print(t5.view(-1, 3, 2).size())
print()

# 對 t5 進行轉置
print(t5.transpose(1, 0))
# 輸出轉置後的維度 (3, 4) 元素的順序會改變
print(t5.transpose(1, 0).is_contiguous())
# reshape 可以對轉置後的 Tensor 進行操作
print(t5.transpose(1, 0).reshape(6, 2))
print()
# view 不能對轉置後的 Tensor 進行操作
try:
    print(t5.transpose(1, 0).view(6, 2))
except Exception as e:
    print(e)
