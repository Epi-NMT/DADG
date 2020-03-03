import torch
from torch import optim, nn


def loss_fn():
    loss_func = torch.nn.CrossEntropyLoss()
    return loss_func

def sgd(model_name, parameters, lr):
    if model_name == 'alexnet':
        opt = optim.SGD(params=parameters, lr=lr, weight_decay=5e-5, momentum=0.9)

    elif model_name == 'resnet18':
        opt = optim.SGD(params=parameters, lr=lr)
    return opt


def lr_scheduler(total_epoch, epoch, lr):
    step_size = int(total_epoch / 2)

    if (epoch+1) <= step_size * 1:
        return lr

    elif(epoch+1) > step_size * 1:
        learning_rate = 5e-6
        return learning_rate


def domain_sgd(model_name, parameters, lr):
    if model_name == 'alexnet':
        opt = optim.SGD(params=parameters, lr=lr)

    elif model_name == 'resnet18':
        opt = optim.SGD(params=parameters, lr=lr)
    return opt

