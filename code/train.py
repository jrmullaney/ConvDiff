from makeDataset import SampleDataset
from convDiff_model import convDiff

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the net, loss function, and optimizer:
net = convDiff()
net.to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(net.parameters())

# Generate and load the dataset
dataset = SampleDataset(n_images=1000, image_size=128, translation=True, vary_psf=False)
loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)

for epoch in range(2):

    running_loss = 0.0
    for i, data in enumerate(loader, 0):
        inputs = data[0].to(device)
        truth = data[1].to(device)
        focus = data[2].to(device)
        focussed_truth = truth[focus]

        # Zero the optimizer, get outputs, calculate loss, backprop and step
        optimizer.zero_grad()
        outputs = net(inputs.float())[focus]
        loss = criterion(outputs, focussed_truth.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 20 == 19:    # print every 20 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0

print('Finished Training')

PATH = './ConvDiff.pth'
torch.save(net.state_dict(), PATH)
