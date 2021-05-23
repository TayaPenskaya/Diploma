import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import cv2
import numpy as np

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.joint_shape = (16, 2)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.head1 = nn.Sequential(
            *block(100, 80),
            *block(80, 64),
            *block(64, 48),
            *block(48, 32)
        )
        
        self.head2 = nn.Sequential(
            *block(19, 16),
            *block(16, 8),
        )
        
        self.tail = nn.Sequential(
            *block(40, 38),
            *block(38, 36),
            *block(36, 34),
            *block(34, 32)
        )

    def forward(self, noise, labels):
        out1 = self.head1(noise)
        labels = labels.type(torch.cuda.FloatTensor)
        out2 = self.head2(labels)
        gen_input = torch.cat((out2, out1), -1)
        joints = self.tail(gen_input)
        joints = joints.view(joints.size(0), *self.joint_shape)
        return joints


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.joint_shape = (16, 2)

        self.model = nn.Sequential(
            nn.Linear(19 + int(np.prod(self.joint_shape)), 24),
            nn.BatchNorm1d(24, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(24, 16),
            nn.BatchNorm1d(16, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(8, 4),
            nn.BatchNorm1d(4, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, joints, labels):
        d_in = torch.cat((joints.view(joints.size(0), -1), labels), -1)
        validity = self.model(d_in)
        return validity
