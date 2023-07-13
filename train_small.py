#from __future__ import absolute_import
from graph_weather import AnalysisDataset, GraphWeatherForecaster
import glob
import numpy as np
import xarray as xr
import torch
from torch.utils.data import DataLoader, Dataset
from graph_weather.models.losses import NormalizedMSELoss
import torch.optim as optim

filepaths = glob.glob("/local/scratch-2/asv34/graph_weather/dataset/one_day/*")

coarsen = 32
ds = AnalysisDataset(filepaths, '/local/scratch-2/asv34/graph_weather/ls_mask.zarr', 0, 0, coarsen)

data = xr.open_zarr(filepaths[0], consolidated=True).coarsen(latitude=coarsen, boundary="pad").mean().coarsen(longitude=coarsen).mean()
lat_lons = np.array(np.meshgrid(data.latitude.values, data.longitude.values)).T.reshape(-1, 2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

criterion = NormalizedMSELoss(lat_lons=lat_lons, feature_variance=[1,1], device=device).to(device)
means = []
dataset = DataLoader(ds, batch_size=1, num_workers=32)

# train_dataset = dataset[:20]
# test_dataset = dataset[20:]

model = GraphWeatherForecaster(lat_lons, num_blocks=2).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.1)

param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print('model size: {:.3f}MB'.format(size_all_mb))

print("Done Setup")
import time

for epoch in range(10):  # loop over the dataset multiple times
    running_loss = 0.0
    test_loss = 0.0
    start = time.time()
    print(f"Start Epoch: {epoch}")
    for i, data in enumerate(dataset):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        print("loaded data")
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        if i < 20: # use first 20 for training and the rest for testing
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            end = time.time()
            print(
                f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i + 1):.3f} Time: {end - start} sec"
            )
        else:
            test_loss += loss.item()
    print(f"test loss after {epoch=} is {test_loss/4}.")

print("Finished Training")