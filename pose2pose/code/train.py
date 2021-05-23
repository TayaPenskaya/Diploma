import argparse
import os
import sys

from utils.config import cfg, cfg_from_file
from data.mpii import MPIIDataset

import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from net import Generator, Discriminator
from matplotlib import pyplot as plt

import numpy as np
import wandb

from datetime import datetime

wandb.init(project='pose2pose')

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=4000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=19, help="number of classes for labels")
parser.add_argument("--n_joints", type=int, default=16, help="number of joints")
parser.add_argument("--dim", type=int, default=2, help="dimension of joints")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

os.makedirs("../data/joints", exist_ok=True) 

dataset = MPIIDataset('../data/mpii.json')
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=True,
)

cuda = True if torch.cuda.is_available() else False

# Loss functions
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# datetime object containing current date and time
now = datetime.now()
dt_string = now.strftime("%d.%m.%Y_%H:%M")
os.makedirs("out/out_%s" % dt_string, exist_ok=True)

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (joints, labels) in enumerate(dataloader):

        batch_size = joints.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_joints = Variable(joints.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))), requires_grad=False)
        #gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

        # Generate a batch of images
        gen_joints = generator(z, labels)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_joints, labels)
        g_loss = torch.sum(-torch.log(validity + 1e-12))
        
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_joints, labels)
        d_real_loss = -torch.log(validity_real + 1e-12)

        # Loss for fake images
        validity_fake = discriminator(gen_joints.detach(), labels)
        d_fake_loss = -torch.log(1 - validity_fake + 1e-12)

        # Total discriminator loss
        d_loss = torch.sum(d_real_loss + d_fake_loss)

        d_loss.backward()
        optimizer_D.step()
        
    wandb.log({"g_loss": g_loss})
    wandb.log({"d_loss": d_loss})
    
    print(gen_joints[0])

    print(
        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
        % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )\
    
    if ((epoch + 1) % 100 == 0):
        torch.save({
            'epoch': epoch,
            'g_state_dict':generator.state_dict(),
            'd_state_dict':discriminator.state_dict(),
            'g_optimizer_state_dict': optimizer_G.state_dict(),
            'd_optimizer_state_dict': optimizer_D.state_dict(),
            }, 'out/out_%s/%d_model.pt' % (dt_string, epoch))
