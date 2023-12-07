import torch


def get_best_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    # 若無 GPU 可用則使用 CPU
    else:
        return torch.device("cpu")
