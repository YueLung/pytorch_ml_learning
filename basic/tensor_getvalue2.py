import torch

# 取連續值

# 宣告 Tensor 變數

t7 = torch.tensor([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

# 輸出張量 t7 位置 0, 1, 2 但是不含位置 3 的值 [0, 10, 20]
print(t7[0:3])
# 輸出張量 t7 位置 7, 8, 9 的值 [70, 80, 90]
print(t7[7:])
# 輸出張量 t7 位置 0, 1 但是不含位置 2 的值 [0, 10]
print(t7[:2])
# 輸出張量 t7 所有位置的值 [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
print(t7[:])
print()

# 宣告 Tensor 變數
t8 = torch.tensor(
    [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11],
    ]
)

# 輸出張量 t8 位置 0, 1, 但是不含位置 2 的值 [[0, 1, 2], [3, 4, 5]]
print(t8[0:2])
print()

# 輸出張量 t8 位置 1, 2, 3 的值 [[3, 4, 5], [6, 7, 8], [9, 10, 11]]
print(t8[1:])
print()

# 輸出張量 t8 位置 0 但是不含位置 1 的值 [[0, 1, 2]]
print(t8[:1])
print()

# 輸出張量 t8 位置 0 但是不含位置 1 的值 [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
print(t8[:])
