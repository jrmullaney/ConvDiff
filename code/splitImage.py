import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as f
import warnings
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class splitImage():

    def __init__(self, image, kernel_size = 256, overlap = 32):
        
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple) and len(kernel_size) == 2:
            self.kernel_size = kernel_size
        else:
            raise ValueError(
                'kernel_size must be int or tuple of length 2'
                )

        if isinstance(overlap, int):
            overlap = (overlap, overlap)
        elif isinstance(overlap, tuple) and len(overlap) == 2:
            pass
        else:
            raise ValueError(
                'overlap must be int or tuple of length 2'
                )

        self.stride = tuple(x - y for x, y in zip(self.kernel_size, overlap))
        self.unfolder = nn.Unfold(
            kernel_size = self.kernel_size, 
            stride = self.stride
            )
        self.checkImage(image)
        self.calcDims()
        self.folder = nn.Fold(
            output_size = self.npix, 
            kernel_size = self.kernel_size,
            stride = self.stride
        )

    def checkImage(self, image):
        
        if isinstance(image, torch.Tensor):
            pass
        elif isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        else:
            raise ValueError(
                'Input image must either be a torch Tensor or numpy Array'
                )

        while len(image.shape) < 4:
            image.unsqueeze_(0)

        if (image.shape[-3] > image.shape[-2] or image.shape[-3] > image.shape[-1]):
            warnings.warn('It looks like the input image may not have the correct dimensions. Ensure: [channels, ydim, xdim]')
        
        self.image = image.to(device)

    def calcDims(self):
        
        nx = (self.image.shape[3] - self.kernel_size[1]) / self.stride[1] + 1
        ny = (self.image.shape[2] - self.kernel_size[0]) / self.stride[0] + 1

        int_nx = (self.image.shape[3] - self.kernel_size[1]) // self.stride[1] + 1
        int_ny = (self.image.shape[2] - self.kernel_size[0]) // self.stride[0] + 1
        
        self.npatches = (
            int_ny + 1 if int_ny != ny else int_ny,
            int_nx + 1 if int_nx != nx else int_nx
            )

        self.npix = (
            (self.npatches[0] - 1) * self.stride[0] + self.kernel_size[0],
            (self.npatches[1] - 1) * self.stride[1] + self.kernel_size[1]
            )

        self.padding = (
            (self.npix[0] - self.image.shape[2]) // 2,
            (self.npix[1] - self.image.shape[3]) // 2
            )

    def split(self):
                
        self.padImage()

        unfolded = self.unfolder(self.image.float())
        
        patches = unfolded.reshape(
            self.image.shape[1], 
            self.kernel_size[0], self.kernel_size[1], 
            self.npatches[0] * self.npatches[1]
            )

        self.patches = patches.permute(3,0,1,2)

    def padImage(self):
        '''
        Function to pad the image so that it can be split into an integer
        number of patches of requested size and overlap. 
        '''

        #Only pad if you have to:
        if (self.npix[0] > self.image.shape[2] or
         self.npix[1] > self.image.shape[2]):
            if device == torch.device("cuda:0"):

                padded = torch.zeros(
                    self.image.shape[0], 
                    self.image.shape[1], 
                    self.npix[0], self.npix[1])

                for i in range(image.shape[1]):
                    self.image[:,i,...] = f.pad(
                        self.image[:,i,...], 
                        (self.padding[1], self.padding[1], 
                        self.padding[0], self.padding[0])
                    )

            else:
                padded = f.pad(
                    self.image, 
                    (self.padding[1], self.padding[1], 
                    self.padding[0], self.padding[0])
                    )
            self.image = padded

    def join(self, patches):

        patches = patches.permute(1, 2, 3, 0)
        prefolded = patches.reshape(
            1, -1, self.npatches[0] * self.npatches[1]
            )

        reconstructed = self.folder(prefolded)
        
        # Divisor sorts out the normalisation; see torch.nn.Unfold documentation
        if device == torch.device('cuda:0'):
            im_ones = torch.cuda.FloatTensor(self.image_shape).fill_(1)
        else:
            im_ones = torch.ones(self.image_shape, dtype = float)
        divisor = self.folder(self.unfolder(im_ones))
        image = reconstructed / divisor
        
        return self.cropImage(image)

    def cropImage(self):

        startx, stopx = self.padding[1], self.npix[1]-self.padding[1]
        starty, stopy = self.padding[0], self.npix[0]-self.padding[0]

        return image[:, :, starty:stopy, startx:stopx]

    

    
