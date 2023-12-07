import torch


print(
    # 確認 torch 的版本
    f'PyTorch version {torch.__version__}\n' +
    # 確認是否有 GPU 裝置
    f'GPU-enabled installation? {torch.cuda.is_available()}'
)