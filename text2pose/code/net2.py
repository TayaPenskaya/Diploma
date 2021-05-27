import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import cv2
import numpy as np

class Generator2(nn.Module):
    def __init__(self):
        super(Generator2, self).__init__()
        
        self.joint_shape = (16, 2)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.head1 = nn.Sequential(
            *block(32, 24),
            *block(24, 16)
        )
        
        self.head2 = nn.Sequential(
            *block(19, 24),
            *block(24, 32)
        )
        
        self.uptail = nn.Sequential(
            *block(48, 64),
            *block(64, 128),
            *block(128, 256)
        )
        
        self.midtail = nn.Sequential(
            *block(256, 512),
            *block(512, 256)
        )
        
        self.downtail = nn.Sequential(
            *block(256, 128),
            *block(128, 64),
            *block(64, 32)
        )

    def forward(self, noise, labels):
        out1 = self.head1(noise)
        labels = labels.type(torch.cuda.FloatTensor)
        out2 = self.head2(labels)
        gen_input = torch.cat((out2, out1), -1)
        
        up_out = self.uptail(gen_input)
        m_out = self.midtail(up_out)
        out = up_out + m_out
        joints = self.downtail(out)

        joints = joints.view(joints.size(0), *self.joint_shape)
        return joints


class Discriminator2(nn.Module):
    def __init__(self):
        super(Discriminator2, self).__init__()

        self.joint_shape = (16, 2)

        self.model = nn.Sequential(
            nn.Linear(19 + int(np.prod(self.joint_shape)), 24),
            nn.BatchNorm1d(24, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(24, 32),
            nn.BatchNorm1d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 4),
            nn.BatchNorm1d(4, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4, 1),
            #nn.Sigmoid()
        )

    def forward(self, joints, labels):
        d_in = torch.cat((joints.view(joints.size(0), -1), labels), -1)
        validity = self.model(d_in)
        return validity

