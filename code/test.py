import numpy as np

from convDiff_model import convDiff
from makeDataset import RealDataset
from splitImage import splitImage

import torch
from torch.utils.data import DataLoader

from astropy.io import fits

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# Load model:
net = convDiff().to(device)
net.load_state_dict(
    torch.load('./TrainedModels/ConvDiff.pth', 
               map_location=device)
    )

hdu = fits.open('../data/test/inj_r255191_2.fits')
image = hdu[1].data
image = np.repeat(image[np.newaxis, :, :], 2, axis=0)
image[1,...] = hdu[2].data
image = torch.from_numpy(image).to(device)

patchsize = 600
overlap = 16
si = splitImage(patchsize, overlap)
patches = si.split(image)

output = torch.zeros_like(patches)

with torch.no_grad():
    for i in range(patches.shape[0]):
        print(i)
        patch = torch.unsqueeze(patches[i,...],0)
        output[i,0,...] = net(patch.float())[0,0,...]

joined = si.join(output)
joined = joined[0,0,...].cpu().numpy()

hdr = fits.Header()
ext0 = fits.PrimaryHDU(hdr)
ext1 = fits.CompImageHDU(joined)
hdul = fits.HDUList([ext0, ext1])
hdul.writeto('../data/test/out_r255191_2.fits', overwrite=True)
