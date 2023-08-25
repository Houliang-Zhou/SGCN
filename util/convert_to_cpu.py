import torch


def cpu(x):
    x = x.to(torch.device("cpu"))

    return (x)