import torch

def gpu_t(x, device):
    x = torch.tensor(x.astype(float)).float()
    y = x.to(device)
    return(y)