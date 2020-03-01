# set up
# import required library

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim # for gradient descent
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

import matplotlib.pyplot as plt # for plotting

torch.manual_seed(1) # set the random seed
class SignClassifier(nn.Module):
    def __init__(self):
        super(SignClassifier, self).__init__()
        self.conv1 = nn.Conv2d(9, 9, 9)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(9, 99, 9)
        self.fc1 = nn.Linear(99 * 99 * 99, 4300)
        self.fc2 = nn.Linear(4300, 43)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 99 * 99 * 99)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

'''
class baseline(nn.Module):
    def __init__(self):
      super(baseline, self).__init__()
      self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 300)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x
        '''

def plot_curve():
    
    return

def train():
    return
