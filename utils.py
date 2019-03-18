import pickle
import random
import string
from datetime import datetime

import torch
import torch.nn as nn


class IncrementalAverage:
    def __init__(self):
        self.value = 0
        self.counter = 0

    def update(self, x):
        self.counter += 1
        self.value += (x - self.value) / self.counter


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class SizePrinter(nn.Module):
    def forward(self, x):
        print(x.size())
        return x


def count_parameters(model, grad_only=True):
    return sum(p.numel() for p in model.parameters() if not grad_only or p.requires_grad)


def to_device(device, *tensors):
    return tuple(x.to(device) for x in tensors)


def loop_iter(iter):
    while True:
        for item in iter:
            yield item


def unique_string():
    return '{}.{}'.format(datetime.now().strftime('%Y%m%dT%H%M%SZ'),
                          ''.join(random.choice(string.ascii_uppercase) for _ in range(4)))


def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pickle_dump(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)
