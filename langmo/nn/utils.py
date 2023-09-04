import torch


def reinit_tensor(tensor):
    mean = tensor.mean().item()
    std = tensor.std().item()
    torch.nn.init.normal_(tensor, mean, std)


def reinit_parameter(param):
    if hasattr(param, "weight"):
        reinit_tensor(param.weight.data)


def reinit_model(model):
    model.apply(reinit_parameter)
