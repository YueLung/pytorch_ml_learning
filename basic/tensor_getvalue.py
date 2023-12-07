import torch

# 張量取值

# 宣告 Tensor 變數
t6 = torch.tensor(
    [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11],
    ]
)

# 輸出張量 t6 中的第 0 個位置的值 [0, 1, 2]
print(t6[0])
# 輸出張量 t6 中的第 1 個位置的值 [3, 4, 5]
print(t6[1])
# 輸出張量 t6 中的第 1 個位置的值 [6, 7, 8]
print(t6[2])
# 輸出張量 t6 中的第 -2 個位置的值 [6, 7, 8]
print(t6[-2])
# 輸出張量 t6 中的第 -1 個位置的值 [9, 10, 11]
print(t6[-1])
print()

# 輸出張量 t6 中的第 [0, 0] 個位置的值 0
print(t6[0, 0])
# 輸出張量 t6 中的第 [0, 1] 個位置的值 1
print(t6[0, 1])
# 輸出張量 t6 中的第 [1, 1] 個位置的值 4
print(t6[1, 1])
# 輸出張量 t6 中的第 [1, 2] 個位置的值 5
print(t6[1, 2])
# 輸出張量 t6 中的第 [-1, -1] 個位置的值 11
print(t6[-1, -1])
# 輸出張量 t6 中的第 [-1, -2] 個位置的值 10
print(t6[-1, -2])
# 輸出張量 t6 中的第 [-2, -1] 個位置的值 8
print(t6[-2, -1])
