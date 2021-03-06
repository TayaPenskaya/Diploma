import argparse
import os
import sys

import time

from utils.config import cfg, cfg_from_file
from data.mpii import MPIIDataset

import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from gan.net import Generator, Discriminator
from matplotlib import pyplot as plt

import numpy as np
import wandb

from datetime import datetime
from utils.loss import get_medians, calc_bones_loss

wandb.init(project='text2pose')

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=48, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-2, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=32, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=2, help="number of classes for labels")
parser.add_argument("--n_joints", type=int, default=16, help="number of joints")
parser.add_argument("--dim", type=int, default=2, help="dimension of joints")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

cat = "lying"

dataset = MPIIDataset('./data/prepared_data/mpii_'+ cat +'.json')

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

#optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=5e-5)
#optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=5e-5)

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def kp2gaussian(kp, spatial_size, kp_variance):
    coordinate_grid = make_coordinate_grid(spatial_size, kp.type())
    number_of_leading_dimensions = len(kp.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = kp.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)
    shape = kp.shape[:number_of_leading_dimensions] + (1, 1, 2)
    kp = kp.view(*shape)
    mean_sub = (coordinate_grid - kp)
    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)
    return out

def make_coordinate_grid(spatial_size, type):
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)
    x = x / (w - 1)
    y = y / (h - 1)
    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)
    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)
    return meshed

def get_model_params_grads(model):
    grads = []
    weights = []
    # add statistics only once(on last step)
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads.append(param.grad.mean().cpu())
        if param.requires_grad:
            weights.append(param.data.mean().cpu())

    return grads, weights

def compute_gradient_penalty(critic, real_data, fake_data, labels, penalty=1, device='cuda'):
    n_elements = real_data.nelement()
    batch_size = real_data.size()[0]
    colors = real_data.size()[1]
    image_width = real_data.size()[2]
    image_height = real_data.size()[3]
    alpha = torch.rand(batch_size, 1).expand(batch_size, int(n_elements / batch_size)).contiguous()
    alpha = alpha.view(batch_size, colors, image_width, image_height).to(device)

    fake_data = fake_data.view(batch_size, colors, image_width, image_height)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)
    critic_interpolates = critic(interpolates, labels)

    gradients = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(critic_interpolates.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * penalty
    return gradient_penalty

# datetime object containing current date and time
now = datetime.now()
dt_string = now.strftime("%d.%m.%Y_%H:%M")
os.makedirs("out/out_%s" % cat, exist_ok=True)

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (joints, labels) in enumerate(dataloader):

        batch_size = joints.shape[0]

        # Adversarial ground truths
        # valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        # fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_joints = Variable(joints.type(FloatTensor))
        real_gaussians = kp2gaussian(real_joints, (64, 64), 5e-4)
        
        # gen_ids = torch.randint(0, opt.n_classes, size = (1, batch_size))
        # gen_labels = torch.nn.functional.one_hot(gen_ids, opt.n_classes)
        # gen_labels = torch.squeeze(gen_labels).cuda()
        labels = Variable(labels.type(LongTensor))
        # one = torch.tensor(1, dtype=torch.float)
        # mone = one * -1
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        for _ in range(5):
            optimizer_D.zero_grad()
            
            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))), requires_grad=False)
            #gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

            # Generate a batch of images
            gen_joints = generator(z, labels)
            gen_gaussians = kp2gaussian(gen_joints, (64, 64), 5e-4)

            # Loss for real images
            validity_real = discriminator(real_gaussians, labels)
            #d_real_loss = torch.log(validity_real + 1e-12)
            d_real_loss = torch.mean(validity_real)
            #d_real_loss.backward(mone)

            # Loss for fake images
            validity_fake = discriminator(gen_gaussians, labels)
            #d_fake_loss = -torch.log(1 - validity_fake + 1e-12)
            d_fake_loss = torch.mean(validity_fake)
            #d_fake_loss.backward(one)

            gradient_penalty = compute_gradient_penalty(discriminator, real_gaussians, gen_gaussians, labels)
            #gradient_penalty.backward()

            # Total discriminator loss
            d_loss = d_fake_loss - d_real_loss + 10*gradient_penalty

            d_loss.backward(retain_graph=True)
            optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------
        
        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))), requires_grad=False)

        # Generate a batch of images
        gen_joints = generator(z, labels)
        gen_gaussians = kp2gaussian(gen_joints, (64, 64), 5e-4)
        
        inverse = torch.sum(torch.abs(torch.sum(gen_joints - torch.roll(gen_joints, 1, 0))) + 1e-9) * 1e6
        inverse_loss = 1 if inverse == 0 else 1/inverse
        
        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_gaussians, labels)

        #bones_loss = torch.mean(torch.sigmoid(FloatTensor(calc_bones_loss(gen_joints.cpu().detach().numpy(), medians))))
        g_loss = torch.sum(torch.nn.functional.relu(-1 * gen_joints))  - torch.mean(validity) + inverse_loss

        g_loss.backward()  
        optimizer_G.step()
        
    g_grads, g_weights = get_model_params_grads(generator)
    d_grads, d_weights = get_model_params_grads(discriminator)

    wandb.log({"g_gradients": wandb.Histogram(g_grads)})
    wandb.log({"d_gradients": wandb.Histogram(d_grads)})
    wandb.log({"g_weights": wandb.Histogram(g_weights)})
    wandb.log({"d_weights": wandb.Histogram(d_weights)})
        
    wandb.log({"g_all_loss": g_loss})
    wandb.log({"g_wass_loss": - torch.mean(validity)})
    wandb.log({"g_neg_coords_loss": torch.sum(torch.nn.functional.relu(-1 * gen_joints))})
    wandb.log({"g_inverse_loss": inverse_loss})
    
    wandb.log({"d_all_loss": d_loss})
    wandb.log({"d_wass_loss": d_fake_loss - d_real_loss})
    wandb.log({"d_real_loss": d_real_loss})
    wandb.log({"d_fake_loss": d_fake_loss})
    wandb.log({"grad_penalty": 10*gradient_penalty})
    
    print(gen_joints[0])

    print(
        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
        % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )\
    
    if ((epoch + 1) % 10 == 0) or ((epoch+1) in [85, 95, 105, 115]):
        torch.save({
            'epoch': epoch,
            'g_state_dict': generator.state_dict(),
            'd_state_dict': discriminator.state_dict(),
            }, 'out/out_%s/%d_model.pt' % (cat, epoch))
