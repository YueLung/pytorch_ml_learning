import torch

# 判斷式取值

# 宣告 Tensor 變數
t11 = torch.tensor([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

# 輸出每個值是否大於 50 的 `torch.Tensor`
print(t11 > 50)
# 輸出 torch.bool
print((t11 > 50).dtype)
# 輸出大於 50 的值 [60, 70, 80, 90]
print(t11[t11 > 50])
# 輸出除以 20 餘數為 0 的值 [0, 20, 40, 60, 80]
print(t11[t11 % 20 == 0])
