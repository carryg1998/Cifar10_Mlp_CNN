r"""
降维的函数详见utils.util.dim_redu函数
"""

import torch
import torchvision
import numpy as np

from PIL import Image

from utils.util import dim_redu

data_path = "./cifar10/"

train_data = torchvision.datasets.CIFAR10(
    root=data_path,
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=False,
)

test_data = torchvision.datasets.CIFAR10(
    root=data_path,
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=False,
)

res = dim_redu(train_data.data, True, 0.5)

print(res.shape)
