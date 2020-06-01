from torch.utils.data import Dataset
import numpy as np
from astropy.modeling.models import Gaussian2D

class SampleDataset(Dataset):
    def __init__(
        self, 
        n_images=200, image_size=256, 
        translation=True, vary_psf=True
        ):
        
        images = np.zeros([n_images,2,image_size,image_size])
        truth = np.zeros([n_images,1,image_size,image_size])
        focus = np.zeros([n_images,1,image_size,image_size], dtype=np.bool)

        images += np.random.normal(size=images.shape)

        y, x = np.mgrid[0:image_size, 0:image_size]
        for i in range(images.shape[0]):
            
            pos = np.random.uniform(low=0, high=image_size-1, size=[6,2])
            intensity = np.random.uniform(low=5, high=15, size=6)
            trans = np.random.normal(0,3,2) if translation else np.zeros(2)
            
            fwhm = np.array([3,3])#)np.random.normal(3,0.75,2)
            fwhm[1] = fwhm[1] if vary_psf else fwhm[0]

            for j in range(pos.shape[0]-1):
                images[i,0,...] += Gaussian2D(
                    intensity[j], 
                    pos[j,0], pos[j,1], 
                    fwhm[0], fwhm[0], 
                    theta=0.5
                    )(x, y)
                images[i,1,...] += Gaussian2D(
                    intensity[j], 
                    pos[j,0] + trans[0], pos[j,1] + trans[1],
                    fwhm[1], fwhm[1], 
                    theta=0.5
                    )(x, y)

            extra = Gaussian2D(
                intensity[5], 
                pos[5,0] + trans[0], pos[5,1] + trans[1],
                fwhm[1], fwhm[1],
                theta=0.5
                )(x, y)
            images[i,1,...] += extra
            
            mask = np.zeros_like(images[1,0,...])
            mask = extra
            truth[i,0,...] = mask

            focus[i,0,...] += extra > intensity[5] / 5.

        self.images = images
        self.truth = truth
        self.focus = focus
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return (
            self.images[idx,:,:,:], 
            self.truth[idx,:,:,:], 
            self.focus[idx,:,:,:]
        )