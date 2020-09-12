from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import nibabel as nib
import numpy as np

class CTSlices(Dataset):
    def __init__(self, ct_dir, mask_dir):
        self.ct_fpaths = sorted(ct_dir.iterdir())
        self.mask_fpaths = sorted(mask_dir.iterdir())
        self.norm = Normalize()
        self.pos = True
        
    def __len__(self):
        return len(self.ct_fpaths)
    
    def __getitem__(self, i):
        ct = nib.load(self.ct_fpaths[i]).get_fdata(dtype=np.float32)
        mask = nib.load(self.mask_fpaths[i]).get_fdata(dtype=np.float32)
        
        ct = self.norm(ct)
        
        if self.pos:
            ct_slice, mask_slice = get_pos_slice(ct, mask)
        else:
            ct_slice, mask_slice = get_neg_slice(ct, mask)
        self.pos = not self.pos
        
        return ct_slice, mask_slice
    
def get_pos_slice(ct, mask):
    for ct_slice, mask_slice in zip(ct.transpose(2, 0, 1), mask.transpose(2, 0, 1)):
        if mask_slice.sum() > 0:
            return ct_slice, mask_slice

def get_neg_slice(ct, mask):
    for ct_slice, mask_slice in zip(ct.transpose(2, 0, 1), mask.transpose(2, 0, 1)):
        if mask_slice.sum() == 0:
            return ct_slice, mask_slice
                                        
            
    
class Normalize:
    def __init__(self):
        self.squash_min = 0
        self.squash_max = 0
        self.orig_mean = 0
        self.orig_std = 0
    
    def __call__(self, x):
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
