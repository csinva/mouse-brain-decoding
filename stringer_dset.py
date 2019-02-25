from torch.utils.data.dataset import Dataset
import scipy.io as sio
from os.path import join as oj
import numpy as np
import torch

class StringerDset(Dataset):
    def __init__(self, data_dir = '/scratch/users/vision/data/neuro_misc/stringer_data_b', crop_center=True, trans=None, downsample=True):
        mt = sio.loadmat(oj(data_dir, 'natimg2800_M160825_MP027_2016-12-14.mat'))
        
        ### stimulus responses
        istim = mt['stim'][0]['istim'][0]   # (n x num_neurons) identities of stimuli in resp        
        resp = mt['stim'][0]['resp'][0]    # (n x num_neurons) stimuli by neurons

#         if train:
#             idxs = (istim <= 2000)
        idxs = (istim < 2801) # 2801 signals gray screen
#             idxs = (istim > 2000) * (istim < 2801) # 2801 signals gray screen
        idxs = idxs.flatten()
        self.istim = istim[idxs].astype(np.int32)
        self.resp = resp[idxs]

        ### loading images
        mt_ims = sio.loadmat(oj(data_dir, 'images_natimg2800_all.mat'))
        if crop_center:        
            imgs = mt_ims['imgs'][:, 90:180, :]  # 68 by 90 by number of images
        else:
            imgs = mt_ims['imgs']  # 68 by 270 by number of images
        self.imgs = imgs.transpose((2, 0, 1)).astype(np.float32) # (n x 68 x 90)
#         print(self.imgs.shape)

        if downsample:
            self.imgs = self.imgs[:, ::2, ::2]

        if trans is not None:
#             self.imgs = self.imgs.reshape(self.imgs.shape[0], 1, self.imgs.shape[1], self.imgs.shape[2])
            self.imgs = trans(imgs)
            print(self.imgs.shape)
    
    # look up image in istim table
    def __getitem__(self, idx):
        return (self.imgs[self.istim[idx] - 1], self.resp[idx])

    def get_all(self):
        return ([self.__getitem])

    def __len__(self):
        return self.istim.shape[0]
    
def get_data():
    # get data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sdset = StringerDset()

    (ims, resps) = sdset[:2300]
    means = np.mean(ims, axis=0)
    stds = np.std(ims, axis=0) + 1e-8 # stds basically just magnifies stuff in the middle, no need to multiply it back
    ims_norm = (ims - means) / stds
    ims = torch.Tensor(ims_norm).to(device)
    # resps = (resps - np.mean(resps, axis=0)) / (np.std(resps, axis=0) + 1e-8)
    resps = torch.Tensor(resps).to(device)


    (ims_val, resps_val) = sdset[2300: 2700]
    means_val = np.mean(ims_val, axis=0)
    stds_val = np.std(ims_val, axis=0) + 1e-8 # stds basically just magnifies stuff in the middle, no need to multiply it back
    ims_norm_val = (ims_val - means_val) / stds_val
    # resps_val = (resps_val - np.mean(resps_val, axis=0)) / (np.std(resps_val, axis=0) + 1e-8)
    ims_val = torch.Tensor(ims_norm_val).to(device)
    resps_val = torch.Tensor(resps_val).to(device)


    return ims, resps, ims_val, resps_val