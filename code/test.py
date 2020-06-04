from skimage import io
import numpy as np
import torch
from splitImage import splitImage
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

im = torch.from_numpy(np.array(io.imread('cat.jpg'))).to(device)

tic = time.process_time()
patches = si.split(im)
joined = si.join(patches)
toc = time.process_time()
print(toc-tic)