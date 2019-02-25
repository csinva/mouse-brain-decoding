import sys
import numpy as np
from os.path import join as oj
import torch
from tqdm import tqdm
from torch.nn import functional as F
from torch import nn, optim
import torchvision.utils as vutils



def get_generator():
    # get gan
    gan_dir = '/accounts/projects/vision/chandan/gan/cifar100_dcgan_grayscale'
    sys.path.insert(1, gan_dir)

    # load the models
    from dcgan import Generator_rect
    num_gpu = 1 if torch.cuda.is_available() else 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    G = Generator_rect(ngpu=num_gpu).to(device)

    # load weights
    G.load_state_dict(torch.load(oj(gan_dir, 'weights_rect/netG_epoch_299.pth'), map_location=device))
    G = G.eval()
    return G

def get_reg_model(lay=1):
    import torchvision.models as tmodels
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vgg = tmodels.vgg19(pretrained=True).eval().to(device)
    if lay == 1:
        reg_model = list(vgg.features.modules())[1] # first lay
    elif lay == 2:
        mods = list(vgg.features.modules())[1: 4]
        mods[1].inplace = False
        reg_model = torch.nn.Sequential(mods[0], mods[1], mods[2])
    return reg_model


class GenNet(nn.Module):
    def __init__(self, G):
        super(GenNet, self).__init__()
        self.fc1 = nn.Linear(11449, 100) # num_neurons to latent space
        self.fc1.weight.data = 1e-3 * self.fc1.weight.data
        self.fc1.bias.data = 1e-3 * self.fc1.bias.data
        self.G = G

    def forward(self, x):
        x = self.fc1(x)
#         print('latent', x[0, :20])
        x = x.reshape(x.shape[0], x.shape[1], 1, 1)
        im = self.G(x)
        return im
    
class LinNet(nn.Module):
    def __init__(self):
        super(LinNet, self).__init__()
        self.fc1 = nn.Linear(11449, 34 * 45) # num_neurons to latent space

    def forward(self, x):
        x = self.fc1(x)
        x = x.reshape(x.shape[0], 34, 45)
        return x