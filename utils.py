import torch
from torch import optim

def loss_fn():
    loss_func = torch.nn.CrossEntropyLoss()
    return loss_func


def sgd(model_name, parameters, lr):
    if model_name == 'alexnet':
        opt = optim.SGD(params=parameters, lr=lr, weight_decay=5e-5, momentum=0.9)

    elif model_name == 'resnet18':
        opt = optim.SGD(params=parameters, lr=lr, weight_decay=5e-5, momentum=0.9)

    return opt

def domain_sgd(model_name, parameters, lr):
    if model_name == 'alexnet':
        opt = optim.SGD(params=parameters, lr=lr)

    elif model_name == 'resnet18':
        opt = optim.SGD(params=parameters, lr=lr)

    return opt




