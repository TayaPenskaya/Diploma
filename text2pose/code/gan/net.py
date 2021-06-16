import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torch.nn.utils import spectral_norm

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
            *block(32, 64),
            *block(64, 128),
        )
        
        self.uptail = nn.Sequential(
            *block(128, 192),
            *block(192, 256),
        )
        
        self.midtail = nn.Sequential(
            *block(256, 384),
            *block(384, 512),
            *block(512, 768),
            *block(768, 512),
            *block(512, 384),
            *block(384, 256),
        )
        
        self.downtail = nn.Sequential(
            *block(256, 192),
            *block(192, 128),
            *block(128, 64),
            *block(64, 32),
        )
        
        self.dropout = nn.Dropout(0.25)

    def forward(self, noise, labels):
        out1 = self.head1(noise)
        #labels = labels.type(torch.cuda.FloatTensor)
        #out2 = self.head2(labels)
        #gen_input = torch.cat((out2, out1), -1)
        
        up_out = self.uptail(out1)
        m_out = self.midtail(up_out)
        m_out = self.dropout(m_out)
        out = up_out + m_out
        joints = self.downtail(out)

        joints = joints.view(joints.size(0), *self.joint_shape)
        return joints


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.head1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State (32x64x64)
            spectral_norm(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State (64x64x64)
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State (128x32x32)
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State (1024x4x4)
            spectral_norm(nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=3, stride=1, padding=1)),)
        
        self.tail = nn.Sequential(
            nn.Linear(16, 4),
            nn.BatchNorm1d(4, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4, 1),
        )
        

    def forward(self, gaussians, labels):
        out1_ = self.head1(gaussians)
        #print('out1_', out1_.size())
        out1 = torch.flatten(out1_, start_dim=1)
        #print('out1', out1.size())
        #labels = labels.type(torch.cuda.FloatTensor)
        #d_in = torch.cat((out1, labels), -1)
        validity = self.tail(out1)
        return out1
