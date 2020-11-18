from torchvision import models
from torch import nn
import torch
from torch.autograd import Function
import os
from skimage import io
from collections import OrderedDict
from itertools import chain
from alexnet import Id

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AlexNetCaffe(nn.Module):
    def __init__(self,  n_classes=100, dropout=True):
        super(AlexNetCaffe, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=4)),
            ("relu1", nn.ReLU(inplace=True)),
            ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm1", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv2", nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
            ("relu2", nn.ReLU(inplace=True)),
            ("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm2", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv3", nn.Conv2d(256, 384, kernel_size=3, padding=1)),
            ("relu3", nn.ReLU(inplace=True)),
            ("conv4", nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
            ("relu4", nn.ReLU(inplace=True)),
            ("conv5", nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
            ("relu5", nn.ReLU(inplace=True)),
            ("pool5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ("fc6", nn.Linear(256 * 6 * 6, 4096)),
            ("relu6", nn.ReLU(inplace=True)),
            ("drop6", nn.Dropout() if dropout else Id()),
            ("fc7", nn.Linear(4096, 4096)),
            ("relu7", nn.ReLU(inplace=True)),
            ("drop7", nn.Dropout() if dropout else Id())]))

        self.class_classifier = nn.Linear(4096, n_classes)

    def get_params(self, base_lr):
        return [{"params": self.features.parameters(), "lr": 0.},
                {"params": chain(self.classifier.parameters(), self.jigsaw_classifier.parameters()
                                 , self.class_classifier.parameters()
                                 ), "lr": base_lr}]

    def is_patch_based(self):
        return False

    def forward(self, x, lambda_val=0):
        x = self.features(x*57.6)  #57.6 is the magic number needed to bring torch data back to the range of caffe data, based on used std
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return self.class_classifier(x)


    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.file.iloc[idx, 0])
        image = io.imread(img_name)
        label = self.label_arr[idx]
        sample = image

        if self.transform:
            sample = self.transform(sample)
        return sample, label

def alexnet(num_class):
    model = AlexNetCaffe(num_class)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, .1)
            nn.init.constant_(m.bias, 0.)

    state_dict = torch.load(os.path.join(os.path.dirname(__file__), "alexnet_caffe.pth.tar"))
    del state_dict["classifier.fc8.weight"]
    del state_dict["classifier.fc8.bias"]
    model.load_state_dict(state_dict, strict=False)
    return model

class NetE_alexnet(nn.Module):

    def __init__(self, classes):
        super(NetE_alexnet, self).__init__()

        net = alexnet(classes)

        # Get the feature extractor
        self.netE = nn.Sequential(*list(net.children())[:-2])

    def forward(self, x, lambda_val=0):
        x = self.netE(x * 57.6)  # 57.6 is the magic number needed to bring torch data back to the range of caffe data, based on used std
        out = x.view(x.size(0), -1)
        return out

class NetC_alexnet(nn.Module):

    def __init__(self, classes):
        super(NetC_alexnet, self).__init__()

        net = alexnet(classes)

        # Get the feature extractor
        self.netC = nn.Sequential(*list(net.children())[-2:])



    def forward(self, input):

        out = self.netC(input)

        return out



def resnet(num_class):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, num_class)
    return model

class NetE_resnet(nn.Module):

    def __init__(self):
        super(NetE_resnet, self).__init__()

        net = models.resnet18(pretrained=True)

        # Get the feature extractor
        self.netE = nn.Sequential(*list(net.children())[:-1])

    def forward(self, input):
        x = self.netE(input)
        out = torch.flatten(x, 1)
        return out

class NetC_resnet(nn.Module):

    def __init__(self, classes):
        super(NetC_resnet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2048, classes)
        )

    def forward(self, input):
        out = self.net(input)
        return out


def nets(model, classes):

    if model == 'alexnet':
        netE = NetE_alexnet(classes)
        netC = NetC_alexnet(classes)
        return netE, netC, NetD_alexnet()

    elif model == 'resnet18':
        netE = NetE_resnet()
        netC = NetC_resnet(classes)
        return netE, netC, NetD_resnet()


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)


class NetD_resnet(nn.Module):
    """AlexNet Discriminator for Meta-Learning Domain Generalization(MLADG) on PACS"""

    def __init__(self, discriminate=1):
        super(NetD_resnet, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, discriminate),
            nn.Sigmoid(),
        )

    def forward(self, input, iter, total_iter):
        """Forward the discriminator with a backward reverse layer"""
        input = GradReverse.grad_reverse(input, constant=1)
        out = self.layer(input)
        return out


class NetD_alexnet(nn.Module):
    """AlexNet Discriminator for Meta-Learning Domain Generalization(MLADG) on PACS"""

    def __init__(self, discriminate=1):
        super(NetD_alexnet, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(9216, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, discriminate),
            nn.Sigmoid(),
        )

    def forward(self, input, iter, total_iter):
        """Forward the discriminator with a backward reverse layer"""
        input = GradReverse.grad_reverse(input, constant=1)
        out = self.layer(input)
        return out

