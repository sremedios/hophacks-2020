import matplotlib.pyplot as plt
import numpy as np
import time
from IPython import display
from PIL import Image

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.cmap'] = 'Greys_r'
plt.rcParams['image.interpolation'] = 'nearest'

def multiplot(imgs, titles=None, vmin=None, vmax=None):
    fig, axs = plt.subplots(1, len(imgs))
    
    if titles is None:
        titles = ['' for _ in range(len(imgs))]

    for ax, img, title in zip(axs, imgs, titles):
        if vmin is None:
            cur_vmin = img.min()
        if vmax is None:
            cur_vmax = img.max()
        ax.imshow(np.rot90(img), vmin=cur_vmin, vmax=cur_vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
    
def center_vol_plot(img_vol, target_shape=None):
    
    cs = [c//2 for c in img_vol.shape]
    
    x = img_vol[cs[0], :, :]
    y = img_vol[:, cs[1], :]
    z = img_vol[:, :, cs[2]]
    
    
    # For anisotropic images, provide `target_shape` for NN interp
    if target_shape is not None:
        # check slice shapes
        if x.shape != (target_shape[1], target_shape[2]):
            # PIL resize needs (y, x)
            x = np.array(Image.fromarray(x)\
                         .resize((target_shape[2], target_shape[1]), 
                                 Image.NEAREST))
        if y.shape != (target_shape[0], target_shape[2]):
            y = np.array(Image.fromarray(y)\
                         .resize((target_shape[2], target_shape[0]), 
                                 Image.NEAREST))
        if z.shape != (target_shape[0], target_shape[1]):
            z = np.array(Image.fromarray(z)\
                         .resize((target_shape[1], target_shape[0]), 
                                 Image.NEAREST))
   
    multiplot(
        [x, y, z],
        ['Sagittal','Coronal','Axial'],
    )