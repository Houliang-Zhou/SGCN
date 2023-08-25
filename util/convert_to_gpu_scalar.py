import torch


def gpu_ts(x, device):
    x = torch.tensor(x).float()
    y = x.to(device)
    return (y)
