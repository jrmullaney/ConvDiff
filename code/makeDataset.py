from torch.utils.data import Dataset
import numpy as np
from astropy.modeling.models import Gaussian2D

class SampleDataset(Dataset):
    def __init__(self, imsize=256):
        images = np.zeros([100,2,imsize,imsize])
        truth = np.zeros([100,1,imsize,imsize])
        
        images += np.random.normal(size=images.shape)

        y, x = np.mgrid[0:imsize, 0:imsize]
        for i in range(images.shape[0]):
            
            pos = np.random.uniform(low=0, high=imsize-1, size=[6,2])
            translation = np.random.normal(0,3,2)
            fwhm = np.random.normal(5,1,2)
            for j in range(pos.shape[0]-1):
                images[i,0,...] += Gaussian2D(
                    5, 
                    pos[j,0], pos[j,1], 
                    fwhm[0], fwhm[0], 
                    theta=0.5
                    )(x, y)
                images[i,1,...] += Gaussian2D(
                    5, 
                    pos[j,0] + translation[0], pos[j,1] + translation[1],
                    fwhm[1], fwhm[1], 
                    theta=0.5
                    )(x, y)
        
            extra = Gaussian2D(
                5, 
                pos[5,0] + translation[0], pos[5,1] + translation[1],
                fwhm[1], fwhm[1],
                theta=0.5
                )(x, y)
            images[i,1,...] += extra
            
            mask = np.zeros_like(images[1,0,...])
            mask = extra
            truth[i,0,...] = mask
        
        self.images = images
        self.truth = truth
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx,:,:,:], self.truth[idx,:,:,:]