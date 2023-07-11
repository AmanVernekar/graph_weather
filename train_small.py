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

ds = AnalysisDataset(filepaths, '/local/scratch-2/asv34/graph_weather/ls_mask.zarr', 0, 0, 1)

data = xr.open_zarr(filepaths[0], consolidated=True)
lat_lons = np.array(np.meshgrid(data.latitude.values, data.longitude.values)).T.reshape(-1, 2)
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
# Get the variance of the variables
criterion = NormalizedMSELoss(lat_lons=lat_lons, feature_variance=[1,1], device=device).to(device)
means = []
dataset = DataLoader(ds, batch_size=1, num_workers=32)
model = GraphWeatherForecaster(lat_lons, feature_dim=597, num_blocks=6).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001)

print("Done Setup")
import time

for epoch in range(10):  # loop over the dataset multiple times
    running_loss = 0.0
    start = time.time()
    print(f"Start Epoch: {epoch}")
    for i, data in enumerate(dataset):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        end = time.time()
        print(
            f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i + 1):.3f} Time: {end - start} sec"
        )

print("Finished Training")