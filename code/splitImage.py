import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as f
import warnings

class splitImage():

    def __init__(self, kernel_size = 256, overlap = 32):
        
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

    def padImage(self, image):
        '''
        Function to pad the image so that it can be split into an integer
        number of patches of requested size and overlap. 
        '''

        image = self.checkImage(image)
        
        nx = (image.shape[3] - self.kernel_size[1]) / self.stride[1] + 1
        ny = (image.shape[2] - self.kernel_size[0]) / self.stride[0] + 1

        int_nx = (image.shape[3] - self.kernel_size[1]) // self.stride[1] + 1
        int_ny = (image.shape[2] - self.kernel_size[0]) // self.stride[0] + 1
        
        new_nx = int_nx + 1 if int_nx != nx else int_nx
        new_ny = int_ny + 1 if int_ny != ny else int_ny

        new_px = (new_nx - 1) * self.stride[1] + self.kernel_size[1]
        new_py = (new_ny - 1) * self.stride[0] + self.kernel_size[0]

        pad_x = (new_px - image.shape[3]) // 2
        pad_y = (new_py - image.shape[2]) // 2

        return f.pad(image, (pad_x, pad_x, pad_y, pad_y))

    def split(self, image):
                
        image = self.padImage(image)

        self.nx = (image.shape[3] - self.kernel_size[1]) // self.stride[1] + 1
        self.ny = (image.shape[2] - self.kernel_size[0]) // self.stride[0] + 1

        unfolded = self.unfolder(image.float())
        
        patches = unfolded.reshape(
            self.image_shape[1], 
            self.kernel_size[0], self.kernel_size[1], 
            self.nx * self.ny
            )

        return patches.permute(3,0,1,2)

    def join(self, patches):

        patches = patches.permute(1, 2, 3, 0)
        prefolded = patches.reshape(1, -1, self.nx * self.ny)

        folder = nn.Fold(
            output_size = self.image_shape[2:], 
            kernel_size = self.kernel_size,
            stride = self.stride
        )
        reconstructed = folder(prefolded)

        # Divisor sorts out the normalisation; see torch.nn.Unfold documentation
        im_ones = torch.ones(self.image_shape, dtype = float)
        divisor = folder(self.unfolder(im_ones))

        return reconstructed / divisor

    def checkImage(self, image):
        
        if isinstance(image, torch.Tensor):
            pass
        elif isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        else:
            raise ValueError(
                'Input image must either be a torch Tensor or numpy Array'
                )

        if (image.shape[2] > image.shape[1] or image.shape[2] > image.shape[0]):
            warnings.warn('It looks like the input image may not have the correct dimensions. Ensure: [ydim, xdim, channels]')

        while len(image.shape) < 4:
            image.unsqueeze_(0)
        
        image = image.permute(0, 3, 1, 2)
        self.image_shape = image.shape

        return image