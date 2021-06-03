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
            *block(64, 80),
            *block(80, 92),
            *block(92, 128),
            *block(128, 144),
        )
        
        self.head2 = nn.Sequential(
            *block(2, 4),
            *block(4, 8),
            *block(8, 16),
            *block(16, 32),
            *block(32, 48),
        )
        
        self.uptail = nn.Sequential(
            *block(192, 224),
            *block(224, 256),
        )
        
        self.midtail = nn.Sequential(
            *block(256, 512),
            *block(512, 640),
            *block(640, 512),
            *block(512, 256),
        )
        
        self.downtail = nn.Sequential(
            *block(256, 192),
            *block(192, 144),
            *block(144, 92),
            *block(92, 64),
            *block(64, 48),
            *block(48, 32),
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

        self.head1 = nn.Sequential(
            nn.Linear(int(np.prod(self.joint_shape)), 32),
            nn.BatchNorm1d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.head2 = nn.Sequential(
            nn.Linear(2, 8),
            nn.BatchNorm1d(8, 0.8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.tail = nn.Sequential(
            nn.Linear(8 + 64, 128),
            nn.BatchNorm1d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 1),
            #nn.Sigmoid()
        )

    def forward(self, joints, labels):
        out_joints = self.head1(joints.view(joints.size(0), -1))
        labels = labels.type(torch.cuda.FloatTensor)
        out_labels = self.head2(labels)
        d_in = torch.cat((out_joints, out_labels), -1)
        validity = self.tail(d_in)
        return validity
