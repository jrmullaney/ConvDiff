import numpy as np

from convDiff_model import convDiff
from makeDataset import RealDataset
from splitImage import splitImage

import torch
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model:
net = convDiff()
net.load_state_dict(
    torch.load('./TrainedModels/ConvDiff.pth', 
               map_location=device
)
patchsize = 512
overlap = 0
dataset = RealDataset('../data/test/inj_r255191_1.fits', patchsize, overlap)
loader = DataLoader(dataset, batch_size=1, num_workers=1)

si = splitImage(patchsize, overlap)
output = torch.zeros([len(loader),0,patchsize,overlap]).to(device)
for i, data in enumerate(loader, 0):
    print(i, len(loader))
    input, tru, mask = data.to(device)
    
    with torch.no_grad():    
        output[i,0,...] = net(input.float())[0,0,...]
si.join(output)
