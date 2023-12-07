import torch

# 如果有可用 GPU 時採用 GPU cuda:0


if torch.cuda.is_available():
    device = torch.device("cuda:0")
# 若無 GPU 可用則使用 CPU
else:
    device = torch.device("cpu")

print(device)

# 根據 device 創造張量
t37 = torch.tensor([1.0, 2.0, 3.0], device=device)
# 使用 to 搬移張量至指定的裝置
t38 = torch.tensor([1.0, 2.0, 3.0]).to(device)
