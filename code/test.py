import numpy as np

from convDiff_model import convDiff
from makeDataset import StarMaker
from splitImage import splitImage

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model:
net = convDiff()
net.load_state_dict(
        torch.load('./ConvDiff.pth',
                                  map_location=torch.device('cpu'))
    )
net.to(device)

fwhm = np.random.normal(3,0.75,2)
sigma = fwhm / 2.35482

sm = StarMaker()
size = (6000,8000)
sm.seedStars(size=size, fraction=1e-3)
ref, refFocus = sm.addPsf(sigma=sigma[0])
sci, sciFocus = sm.addPsf(sigma=sigma[1])
sm.seedStars(size=size, fraction=1e-4)
trans, transFocus = sm.addPsf(sigma=sigma[1])

input = np.repeat(ref[np.newaxis, :, :], 2, axis=0)
input[1,...] = sci + trans

tic = time.process_time()
si = splitImage(512, 32)
patches = si.split(input).to(device)
output = torch.zeros_like(patches[:,:,:,:])

for i in range(patches.shape[0]):
    with torch.no_grad():
        output[i,0,...] = net(patches[i,...].unsqueeze_(0).float())[0,...]
joined = si.join(output)
print(joined[0,0,0,0])
toc = time.process_time()
print(toc-tic)
