from torch.utils.data.dataset import Dataset
import scipy.io as sio
from os.path import join as oj
import numpy as np

class StringerDset(Dataset):
    def __init__(self, data_dir = '/scratch/users/vision/data/neuro_misc/stringer_data_b', train=True, crop_center=True, trans=None):
        mt = sio.loadmat(oj(data_dir, 'natimg2800_M160825_MP027_2016-12-14.mat'))

        ### stimulus responses
        istim = mt['stim'][0]['istim'][0]   # (n x num_neurons) identities of stimuli in resp        
        resp = mt['stim'][0]['resp'][0]    # (n x num_neurons) stimuli by neurons

        if train:
            idxs = (istim <= 2000)
        else:
            idxs = (istim > 2000) * (istim < 2801) # 2801 signals gray screen
        idxs = idxs.flatten()
        self.istim = istim[idxs].astype(np.int32)
        self.resp = resp[idxs]

        ### loading images
        mt_ims = sio.loadmat(oj(data_dir, 'images_natimg2800_all.mat'))
        if crop_center:        
            imgs = mt_ims['imgs'][:, 90:180, :]  # 68 by 90 by number of images
        else:
            imgs = mt_ims['imgs']  # 68 by 270 by number of images
        self.imgs = imgs.transpose((2, 0, 1)).astype(np.float32)
#         print(self.imgs.shape)
        if trans is not None:
#             self.imgs = self.imgs.reshape(self.imgs.shape[0], 1, self.imgs.shape[1], self.imgs.shape[2])
            self.imgs = trans(imgs)
            print(self.imgs.shape)
    
    # look up image in istim table
    def __getitem__(self, idx):
        return (self.imgs[self.istim[idx] - 1], self.resp[idx])

    def __len__(self):
        return self.istim.shape[0]