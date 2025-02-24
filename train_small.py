#from __future__ import absolute_import
from graph_weather import AnalysisDataset, GraphWeatherForecaster
import glob
import numpy as np
import xarray as xr
import torch
from torch.utils.data import DataLoader, Dataset
from graph_weather.models.losses import NormalizedMSELoss
import torch.optim as optim

ds = AnalysisDataset('/local/scratch-2/asv34/graph_weather/dataset/jan_2022_normed.npy')
filepaths = glob.glob("/local/scratch-2/asv34/graph_weather/dataset/2022/*")
dataset = DataLoader(ds, batch_size=1, num_workers=32)
coarsen = 8 # change this in preprocessor too if changed here

data = xr.open_zarr(filepaths[0], consolidated=True).coarsen(latitude=coarsen, boundary="pad").mean().coarsen(longitude=coarsen).mean()
lat_lons = np.array(np.meshgrid(data.latitude.values, data.longitude.values)).T.reshape(-1, 2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

criterion = NormalizedMSELoss(lat_lons=lat_lons, feature_variance=[1,1], device=device).to(device)
means = []


# train_dataset = dataset[:20]
# test_dataset = dataset[20:]
train_count = 100
test_count = len(ds) - train_count

model = GraphWeatherForecaster(lat_lons, feature_dim=42, num_blocks=6).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-5)

param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print('model size: {:.3f}MB'.format(size_all_mb))

print("Done Setup")
# import time

for epoch in range(100):  # loop over the dataset multiple times
    running_loss = 0.0
    test_loss = 0.0
    # start = time.time()
    # print(f"Start Epoch: {epoch+1}")
    for i, data in enumerate(dataset):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].float().to(device), data[1].float().to(device)
        if i < train_count: # use first 100 for training and the remaining 23 for testing
            model.train()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            # end = time.time()

            # print(
            #     f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i + 1):.3f} Time: {end - start} sec"
            # )
        else:
            model.eval()
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
    
    print(f"train loss after epoch {epoch+1} is {running_loss/train_count}.")
    print(f"test loss after epoch {epoch+1} is {test_loss/test_count}.")

print("Finished Training")
torch.save(model.state_dict(), '/local/scratch-2/asv34/graph_weather/dataset/models/jan2022_rescaled_100epochs.pt')