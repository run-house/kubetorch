def torch_summer(a, b):
    import torch

    res = int(torch.sum(torch.tensor([a, b])))
    return res
