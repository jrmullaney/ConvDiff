import numpy as np
import torch
import torch.nn as nn
import warnings

class splitImage():

    def __init__(self, kernel_size = 256, overlap = 32):
        self.kernel_size = kernel_size
        self.stride = kernel_size - overlap

    def split(self, image):
                
        if isinstance(image, torch.Tensor):
            pass
        elif isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        else:
            raise ValueError(
                'Input image must either be a torch Tensor or numpy Array'
                )

        if (image.shape[2] > image.shape[1] or image.shape[2] > image.shape[0]):
            warnings.warn('It looks like the input image may not have the correct dimensions. Check it is: [ydim, xdim, channels]')

        while len(image.shape) < 4:
            image.unsqueeze_(0)
        image = image.permute(0, 3, 1, 2)
        self.image_shape = image.shape

        self.nx = int((image.shape[3] - self.kernel_size) / self.stride + 1)
        self.ny = int((image.shape[2] - self.kernel_size) / self.stride + 1)

        self.unfolder = nn.Unfold(
            kernel_size = self.kernel_size, 
            stride = self.stride
            )
        unfolded = self.unfolder(image.float())
        
        patches = unfolded.reshape(
            self.image_shape[1], 
            self.kernel_size, self.kernel_size, 
            self.nx * self.ny
            )

        return patches.permute(3,0,1,2)

    def join(self, patches):

        patches = patches.permute(1, 2, 3, 0)
        prefolded = patches.reshape(1, -1, self.nx * self.ny)

        self.folder = nn.Fold(
            output_size = self.image_shape[2:], 
            kernel_size = self.kernel_size,
            stride = self.stride
        )
        reconstructed = self.folder(prefolded)

        # Divisor sorts out the normalisation; see torch.nn.Unfold documentation
        im_ones = torch.ones(self.image_shape, dtype = float)
        divisor = self.folder(self.unfolder(im_ones))

        return reconstructed / divisor