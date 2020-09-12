from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import nibabel as nib
import numpy as np

class CTSlices(Dataset):
    def __init__(self, ct_dir, mask_dir):
        self.ct_fpaths = sorted(ct_dir.iterdir())
        self.mask_fpaths = sorted(mask_dir.iterdir())
        self.norm = Normalize(hu_min=-1000, hu_max=100)
        self.pos = True
        
    def __len__(self):
        return len(self.mask_fpaths)
    
    def __getitem__(self, i):
        ct = nib.load(self.ct_fpaths[i]).get_fdata(dtype=np.float32)
        mask = nib.load(self.mask_fpaths[i]).get_fdata(dtype=np.float32)
        
        ct = self.norm(ct)
        ct_slice, mask_slice = get_slice(ct, mask, self.pos)
#         self.pos = not self.pos
        
        return ct_slice.cuda(), mask_slice.cuda()
    
def get_slice(ct, mask, is_pos):
    i = np.random.randint(0, ct.shape[-1])
    
    if is_pos:
        while mask[..., i].sum() == 0:
            i = np.random.randint(0, mask.shape[-1])
    else:
        while mask[..., i].sum() != 0:
            i = np.random.randint(0, mask.shape[-1])
            
    ct = torch.from_numpy(ct[np.newaxis, ..., i]).float()
    mask = torch.from_numpy(mask[np.newaxis, ..., i]).float()
    return ct, mask


class Normalize:
    def __init__(self, hu_min, hu_max):
        self.squash_min = 0
        self.squash_max = 0
        self.orig_mean = 0
        self.orig_std = 0
        self.hu_min = hu_min
        self.hu_max = hu_max
    
    def __call__(self, x):
        x[np.where(x < self.hu_min)] = self.hu_min
        x[np.where(x > self.hu_max)] = self.hu_max
        self.orig_mean = x.mean()
        self.orig_std = x.std()
        x = (x - self.orig_mean) / self.orig_std
        self.squash_min = x.min()
        self.squash_max = x.max()
        x = (x - self.squash_min) / (self.squash_max - self.squash_min)
        return x
    
    def __inv__(self, x):
        term1 = self.squash_min
        term2 = (self.squash_max - self.squash_min)
        return (term1 + (term2 * x)) * self.orig_std + self.orig_mean
