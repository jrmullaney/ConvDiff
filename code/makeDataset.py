from torch.utils.data import Dataset
import numpy as np
import math
from splitImage import splitImage
import torch

from astropy.convolution import Kernel2D, convolve
from astropy.modeling import models
from astropy.io import fits

import glob
from os.path import join
import re
import time

class Gaussian2DKernel(Kernel2D):
    
    '''
    Re-define Gaussian2DKernel to allow positional offsets
    '''

    #_separable = True
    #_is_bool = False

    def __init__(self, x_stddev, y_stddev=None, theta=0.0, x_pos=0.0, y_pos=0.0, **kwargs):
        if y_stddev is None:
            y_stddev = x_stddev
        self._model = models.Gaussian2D(1. / (2 * np.pi * x_stddev * y_stddev),
                                        x_pos, y_pos, x_stddev=x_stddev,
                                        y_stddev=y_stddev, theta=theta)
        self._default_size = self._round_up_to_odd_integer(
            8 * np.max([x_stddev, y_stddev]))
        super().__init__(**kwargs)
        self._truncation = np.abs(1. - self._array.sum())

    @staticmethod
    def _round_up_to_odd_integer(value):
        i = math.ceil(value)
        if i % 2 == 0:
            return i + 1
        else:
            return i


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
        intensity = np.random.uniform(low=50, high=150, size=nstars)
        
        self.positionMask[mask] = 1.
        self.pointSources[mask] = intensity

    def addPsf(self, sigma=3, translation=np.array([0,0])):

        if isinstance(translation, np.ndarray) == False or translation.shape != (2,):
            raise ValueError("translate must be a 2-element numpy array: [x-shift, y-shift]")

        kernel = Gaussian2DKernel(
            x_stddev=sigma, 
            x_pos=translation[0], y_pos=translation[1]
            )
        stars = convolve(self.pointSources, kernel)
        mask = convolve(self.positionMask, kernel)
        mask = mask > 0.2 * mask.max()

        focus = np.zeros_like(stars)

        return stars, mask

class SampleDataset(Dataset):
    def __init__(
        self, 
        n_images = 200, image_size = (1200, 1800),
        patch_size = 512, overlap = 54,
        translate=True, vary_psf=True,
        ):
        
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        elif isinstance(image_size, tuple) and len(image_size) == 2:
            pass
        else:
            raise ValueError(
                'image_size must either be an int or 2-value tuple.'
                )

        image = np.zeros([2,image_size[0],image_size[1]])
        truth = np.zeros([1,image_size[0],image_size[1]])
        focus = np.zeros([1,image_size[0],image_size[1]])

        si = splitImage(kernel_size = patch_size, overlap = overlap)
        patches = si.split(image)
        n_patches = patches.shape[0]

        patch_image = np.zeros([n_images * n_patches, 2, patch_size, patch_size])
        patch_truth = np.zeros([n_images * n_patches, 1, patch_size, patch_size])
        patch_focus = np.zeros([n_images * n_patches, 1, patch_size, patch_size])

        sm = StarMaker()

        for i in range(n_images):
                        
            fwhm = np.random.normal(3,0.75,2)
            fwhm[1] = fwhm[1] if vary_psf else fwhm[0]
            sigma = fwhm / 2.35482

            translation = np.random.uniform(low=-3, high=3, size=2) if translate else np.array([0.,0.])

            sm.seedStars(size=image_size, fraction=1e-3)
            ref, refFocus = sm.addPsf(sigma=sigma[0]) 
            sci, sciFocus = sm.addPsf(sigma=sigma[1],translation=translation)
            
            sm.seedStars(size=image_size, fraction=1e-4)
            trans, transFocus = sm.addPsf(sigma=sigma[1],translation=translation)

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

class RealDataset(Dataset):

    def __init__(
        self,
        file_path = None, 
        patch_size = 512, overlap = 54,
        ):

        files = glob.glob(join(file_path))
        n_files = len(files)

        hdu = fits.open(files[0])
        image = hdu[1].data
        images = np.zeros([5, image.shape[0], image.shape[1]])

        si = splitImage(kernel_size = patch_size, overlap = overlap)
        patches = si.split(image)    
        n_patches = patches.shape[0]

        patches_all = np.zeros([n_files * n_patches, 4, patch_size, patch_size])

        for i, file in enumerate(files):
            
            tic = time.process_time()
            hdu = fits.open(file)
            ref = hdu[2].data
            images[0,...] = hdu[1].data
            images[1,...] = hdu[3].data
            images[3,...] = hdu[3].data
            images[4,...] = hdu[4].data
            hdu.close()
            
            toc = time.process_time()
            print('Reading:', toc-tic)

            images = si.padImage(images)
            ref = images[0,1,...]
            x = torch.arange(images.shape[2], dtype=int)
            y = torch.arange(images.shape[3], dtype=int)
            xv, yv = torch.meshgrid(x, y)
            images[0,1,...] = yv
            images[0,2,...] = xv
            
            tic = time.process_time()
            print('Padding:', tic-toc)

            split = si.split(images)
            toc = time.process_time()
            print('Splitting:', toc-tic)

            xind = split[:,1,...].long()
            yind = split[:,2,...].long()
            split[:,1,...] = ref[0,0,yind,xind]

            tic = time.process_time()
            print('Referencing:', tic-toc)

            patch_ind = np.arange(n_patches * i,n_patches * (i+1), dtype=int)
            patches_all[patch_ind,0,:,:] = split[:,0,:,:]
            patches_all[patch_ind,1,:,:] = split[:,1,:,:]
            patches_all[patch_ind,2,:,:] = split[:,3,:,:]
            patches_all[patch_ind,3,:,:] = split[:,4,:,:]

        self.image = patches_all[:,0:1,...]
        #self.truth = patch_truth[:,2,...]
        #self.focus = patch_focus[:,3,...]
    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, idx):
        return (
            self.image[idx,:,:,:], 
            self.truth[idx,:,:,:], 
            self.focus[idx,:,:,:]
        )