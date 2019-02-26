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
parser.add_argument('--reg1', type=float, default=0.0, help='amount to depend on lay1 nn reg')
parser.add_argument('--reg2', type=float, default=0.0, help='amount to depend on lay2 nn reg')
args = parser.parse_args()


# hyperparams
learning_rate = args.lr
lambda_reg1 = args.reg1
lambda_reg2 = args.reg2
print('lambda_reg1', lambda_reg1, 'lr', learning_rate, 'lambda_reg2', lambda_reg2)
out_dir = '/scratch/users/vision/chandan/decoding/' + 'reg1=' + str(lambda_reg1) + 'reg2=' + str(lambda_reg2) + '_lr=' + str(learning_rate)
# out_dir = 'test'
its = 60000
num_gpu = 1 if torch.cuda.is_available() else 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# data/model
ims, resps, ims_val, resps_val = stringer_dset.get_data()
G = models.get_generator()
reg_model1 = models.get_reg_model(lay=1) # 1 or 'all' supported
reg_model2 = models.get_reg_model(lay=2) # 1 or 'all' supported

# optimization
loss_fn = torch.nn.MSELoss(reduction='sum')
model = models.GenNet(G).to(device)
optimizer = torch.optim.SGD(model.fc1.parameters(), 
                            lr=learning_rate)

# loss/saving
os.makedirs(out_dir, exist_ok=True)
divisor = 34 * 45 * resps.shape[0]
val_loss_best = 1e5

print('training...')        
for it in range(its):
    # lr step down
    if it == 100:
        optimizer.param_groups[0]['lr'] *= 0.1
    if it == 600:
        optimizer.param_groups[0]['lr'] *= 0.5
    if it == 1000:
        optimizer.param_groups[0]['lr'] *= 0.25    
    if it == 15000:
        optimizer.param_groups[0]['lr'] *= 0.5    
    if it == 50000:
        optimizer.param_groups[0]['lr'] *= 0.5        
    
    ims_pred = model(resps)
    loss = loss_fn(ims_pred, ims)
    if lambda_reg1 > 0:
        loss = loss + lambda_reg1 * resps.shape[0] * 1 - utils.lay_sim(reg_model1, ims, ims_pred)
    if lambda_reg2 > 0:
        loss = loss + lambda_reg2 * resps.shape[0] * 1 - utils.lay_sim(reg_model2, ims, ims_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if torch.sum(model.fc1.weight.grad).detach().item() == 0:
        print('zero grad!')
        print('w', torch.sum(model.fc1.weight))    
        break

    if it % 100 == 0:
        with torch.no_grad():
            # training
            loss_mse = loss_fn(ims_pred, ims).detach().item() / divisor
            loss_reg1, loss_reg2 = 0, 0
            if lambda_reg1 > 0:
                loss_reg1 = 1 - utils.lay_sim(reg_model1, ims_pred, ims).detach().item()
                loss = loss_mse + lambda_reg1 * loss_reg1
            if lambda_reg2 > 0:
                loss_reg2 = 1 - utils.lay_sim(reg_model2, ims_pred, ims).detach().item()
                loss = loss_mse + lambda_reg2 * loss_reg2
            lr = optimizer.param_groups[0]['lr']
            print(f'train mse {loss_mse:0.4f} reg1 {loss_reg1:0.4f} reg2 {loss_reg2:0.4f} lr {lr}')


            ims_pred_val = model(resps_val)
            val_loss_mse = loss_fn(ims_pred_val, ims_val).detach().item() / (34 * 45 * resps_val.shape[0])
            val_loss, val_loss_reg1, val_loss_reg2 = 0, 0, 0
            if lambda_reg1 > 0:
                val_loss_reg1 = 1 - utils.lay_sim(reg_model1, ims_pred_val, ims_val).detach().item()
                val_loss = val_loss_mse + lambda_reg1 * val_loss_reg1
            if lambda_reg2 > 0:
                val_loss_reg2 = 1 - utils.lay_sim(reg_model2, ims_pred_val, ims_val).detach().item()
                val_loss = val_loss_mse + lambda_reg2 * val_loss_reg2

            print(f'val mse {val_loss_mse:0.4f} reg1 {val_loss_reg1:0.4f} reg2 {val_loss_reg2:0.4f}')
            if val_loss < val_loss_best:
                val_loss_best = val_loss
                utils.save_ims(out_dir, ims_pred, ims, it, num_ims=50, loss=loss)
                utils.save_ims(out_dir, ims_pred_val, ims_val, it, num_ims=50, val=True, loss=val_loss)
                if it % 1000 == 0:
                    torch.save(model.state_dict(), oj(out_dir, 'model_' + str(it) + '.pth'))