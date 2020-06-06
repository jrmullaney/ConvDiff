from torch.utils.data import Dataset
import numpy as np
from splitImage import splitImage

from astropy.convolution import Gaussian2DKernel, convolve
from astropy.convolution import convolve

class StarMaker():
    def __init__(self):
        pass

    def seedStars(self, size = 512, fraction=0.001):
        '''
        Popululate an image with 2D gaussians to represent stars.
        '''

        if isinstance(size, int):
            size = (size, size)
        elif isinstance(size, tuple) and len(size) == 2:
            pass
        else:
            raise ValueError(
                'size must either be an int or 2-value tuple.'
                )

        if fraction <= 0. or fraction >=1:
             raise ValueError(
                'fraction must be a float between 0 and 1.'
                )

        self.pointSources = np.zeros([size[0], size[1]])
        self.positionMask = np.zeros([size[0], size[1]])
        
        rand = np.random.uniform(low=0., high=1., size=size) 
        mask = np.zeros([size[0], size[1]], dtype=bool)
        
        i = 0
        while np.all(mask == False):
            mask = rand > (1. - fraction)
            i += 1
            if i == 10:
                raise RuntimeError('Fraction too low. No stars generated after ten attempts')

        nstars = mask.sum()
        intensity = np.random.uniform(low=5, high=15, size=nstars)
        
        self.positionMask[mask] = 1.
        self.pointSources[mask] = intensity

    def addPsf(self, sigma=3):

        kernel = Gaussian2DKernel(x_stddev=sigma)
        stars = convolve(self.pointSources, kernel)
        mask = convolve(self.positionMask, kernel)
        mask = mask > 0.2 * mask.max()

        focus = np.zeros_like(stars)

        return stars, mask

class SampleDataset(Dataset):
    def __init__(
        self, 
        n_images=200, image_size=1200, 
        translation=True, vary_psf=True
        ):
        
        image = np.zeros([2,image_size,image_size])
        truth = np.zeros([1,image_size,image_size])
        focus = np.zeros([1,image_size,image_size])

        si = splitImage(kernel_size=512, overlap=64)
        patches = si.split(image)
        n_patches = patches.shape[0]

        patch_image = np.zeros([n_images * n_patches, 2, 512, 512])
        patch_truth = np.zeros([n_images * n_patches, 1, 512, 512])
        patch_focus = np.zeros([n_images * n_patches, 1, 512, 512])

        sm = StarMaker()

        for i in range(n_images):
                        
            fwhm = np.array([5,5])#)np.random.normal(3,0.75,2)
            fwhm[1] = fwhm[1] if vary_psf else fwhm[0]
            sigma = fwhm / 2.35482

            sm.seedStars(size=image_size, fraction=1e-3)
            ref, refFocus = sm.addPsf(sigma=sigma[0]) 
            sci, sciFocus = sm.addPsf(sigma=sigma[1])
            
            sm.seedStars(size=image_size, fraction=1e-4)
            trans, transFocus = sm.addPsf(sigma=sigma[1])


            image[0,...] += ref
            image[1,...] += sci + trans
            truth[0,...] = trans
            focus[0,sciFocus] = 1.
            focus[0,transFocus] = 2. 

            image += np.random.normal(size=image.shape)

            patch_image[n_patches * i:n_patches * (i+1),:,:,:] = si.split(image)
            patch_truth[n_patches * i:n_patches * (i+1),:,:,:] = si.split(truth)
            patch_focus[n_patches * i:n_patches * (i+1),:,:,:] = si.split(focus)

            image.fill(0.)
            truth.fill(0.)
            focus.fill(0)

        self.image = patch_image
        self.truth = patch_truth
        self.focus = patch_focus
    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, idx):
        return (
            self.image[idx,:,:,:], 
            self.truth[idx,:,:,:], 
            self.focus[idx,:,:,:]
        )