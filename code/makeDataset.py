from torch.utils.data import Dataset
import numpy as np
from astropy.modeling.models import Gaussian2D

class SampleDataset(Dataset):
    def __init__(self):
        images = np.zeros([100,2,256,256])
        truth = np.zeros([100,1,256,256])
        
        images += np.random.normal(size=images.shape)

        y, x = np.mgrid[0:256, 0:256]
        for i in range(images.shape[0]):
            
            data = np.zeros([256,256])
            
            pos = np.random.uniform(low=0, high=255, size=[6,2])
            for j in range(pos.shape[0]-1):
                data = data + Gaussian2D(5, pos[j,0], pos[j,1], 5, 5, theta=0.5)(x, y)
    
            images[i,0,...] += data
    
            extra = Gaussian2D(5, pos[5,0], pos[5,1], 5, 5, theta=0.5)(x, y)
            images[i,1,...] += data + extra
            
            mask = np.zeros_like(data)
            mask = extra
            truth[i,0,...] = mask
        
        self.images = images
        self.truth = truth
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx,:,:,:], self.truth[idx,:,:]