import sys
import numpy as np
from os.path import join as oj
import torch
from tqdm import tqdm
from torch.nn import functional as F
from torch import nn, optim
import torchvision.utils as vutils
    
def viz_ims(ims_pred, ims, num_ims=5):    
    plt.figure(figsize=(num_ims * 1.2, 2), dpi=100)
    R, C = 2, num_ims
    for i in range(num_ims):
        plt.subplot(R, C, i + 1)
        plt.imshow(ims_pred[i].cpu().detach().numpy().reshape(34, 45), interpolation='bilinear', cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(hspace=0, wspace=0, left=0)
    for i in range(num_ims):
        plt.subplot(R, C, i + 1 + num_ims)
        plt.imshow(ims[i].cpu().detach().numpy().reshape(34, 45), interpolation='bilinear', cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(hspace=0, wspace=0, left=0)
    plt.show()

def save_ims(out_dir, ims_pred, ims, it, num_ims=5, val=False):      
    suffix = '_val' if val else ''
    
    ims_save = np.empty((2 * num_ims, 1, 34, 45), dtype=np.float32)
    ims = ims[:num_ims].cpu().detach().numpy()
    ims -= np.min(ims, axis=0)
    ims /= np.max(ims, axis=0)
    ims_save[0::2] = ims
    
    ims_pred = ims_pred[:num_ims].cpu().detach().numpy()
    ims_pred -= np.min(ims_pred, axis=0)
    ims_pred /= np.max(ims_pred, axis=0)
    ims_save[0::2] = ims
    ims_save[1::2] = ims_pred
    ims_save = torch.Tensor(ims_save)
    vutils.save_image(ims_save,
                '{}/{}_samples{}.png'.format(out_dir, it, suffix),
                normalize=False, nrow=10)  
    
# vgg 
def lay1_sim(reg_model, im1, im2):
    # grayscale to 3 channel
    
    im1 = im1.expand(-1, 3, -1, -1)
    im2 = im2.expand(-1, 3, -1, -1)
    
    feat1 = reg_model(im1).flatten()
    feat2 = reg_model(im2).flatten()
    feat1 = feat1 / feat1.norm()
    feat2 = feat2 / feat2.norm()
    return torch.dot(feat1, feat2)