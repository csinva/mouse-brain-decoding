import sys
import numpy as np
from os.path import join as oj
import torch
from tqdm import tqdm
from torch.nn import functional as F
from torch import nn, optim
import torchvision.utils as vutils
import stringer_dset
import models
import utils    
import argparse
import os

parser = argparse.ArgumentParser(description='parameters for training decoder')
parser.add_argument('--lr', type=float, default=1e-11, help='learning rate')
parser.add_argument('--reg', type=float, default=0.0, help='amount to depend on nn reg')
args = parser.parse_args()
learning_rate = args.lr
lambda_reg = args.reg
print('lambda_reg', lambda_reg, 'lr', learning_rate)
out_dir = '/scratch/users/vision/chandan/decoding/' + 'lambda=' + str(lambda_reg) + '_lr=' + str(learning_rate)
its = 10000
num_gpu = 1 if torch.cuda.is_available() else 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'


ims, resps, ims_val, resps_val = stringer_dset.get_data()
G = models.get_generator()
reg_model = models.get_reg_model()
os.makedirs(out_dir, exist_ok=True)
loss_fn = torch.nn.MSELoss(reduction='sum')
model = models.GenNet(G).to(device)
optimizer = torch.optim.SGD(model.fc1.parameters(), 
                            lr=learning_rate)
divisor = 34 * 45 * resps.shape[0]

print('training...')        
for it in range(its):
    # lr step down
    if it == 100:
        optimizer.param_groups[0]['lr'] *= 0.1
    if it == 600:
        optimizer.param_groups[0]['lr'] *= 0.5
    if it == 1000:
        optimizer.param_groups[0]['lr'] *= 0.25    
    if it == 20000:
        optimizer.param_groups[0]['lr'] *= 0.5    
    if it == 50000:
        optimizer.param_groups[0]['lr'] *= 0.5        
    
    ims_pred = model(resps)
    loss = loss_fn(ims_pred, ims) + lambda_reg * 1 - utils.lay1_sim(reg_model, ims, ims_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if it % 20 == 0:
        print(it, 'loss', loss.detach().item() / divisor, 'lr', optimizer.param_groups[0]['lr'])
    if torch.sum(model.fc1.weight.grad).detach().item() == 0:
        print('zero grad!')
        print('w', torch.sum(model.fc1.weight))    
        break

    if it % 100 == 0:
        utils.save_ims(out_dir, ims_pred, ims, it, num_ims=50)
        print('\tloss mse', loss_fn(ims_pred, ims).detach().item() / divisor)
        print('\tloss reg', 1 - utils.lay1_sim(reg_model, ims_pred, ims).detach().item())
        with torch.no_grad():
            ims_pred_val = model(resps_val)
            utils.save_ims(out_dir, ims_pred_val, ims_val, it, num_ims=50, val=True)
            print('\tval loss mse', loss_fn(ims_pred_val, ims_val).detach().item() / (34 * 45 * resps_val.shape[0]))
            print('\tval loss reg', 1 - utils.lay1_sim(reg_model, ims_pred_val, ims_val).detach().item())
    if it % 1000 == 0:
        torch.save(model.state_dict(), oj(out_dir, 'model_' + str(it) + '.pth'))