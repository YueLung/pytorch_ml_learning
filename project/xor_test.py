import torch

from module.xor.model import XorModel


xorModel = XorModel()
xorModel.load_state_dict(torch.load("./project/module/xor/model.ckpt"))

inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)

with torch.no_grad():
    pred = (xorModel(inputs) > 0.5).float()
    # pred = xorModel(inputs)

    # print(f"pred = {pred}")

    for input_data, prediction in zip(inputs, pred):
        print(f"{input_data.cpu().numpy()}\t\t{prediction.item()}")
    # for input in inputs:
    #     pred = xorModel(input)

    #     print(f"input = {input} pred = {pred}")
