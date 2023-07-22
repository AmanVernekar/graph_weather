#from __future__ import absolute_import
from graph_weather import AnalysisDataset, GraphWeatherForecaster, ParallelDataset, ParallelForecaster
import glob
import numpy as np
import xarray as xr
import torch
from torch.utils.data import DataLoader, Dataset
from graph_weather.models.losses import NormalizedMSELoss
import torch.optim as optim

num_steps = 3

# ds = AnalysisDataset(file_name)
# ds = ParallelDataset(np_file=file_name, num_steps=num_steps)
# ds_list = [ParallelDataset(np_file=f'/local/scratch-2/asv34/graph_weather/dataset/2022_{month}_normed.npy', num_steps=num_steps) for month in [1,4,7,10]]
ds_list = [AnalysisDataset(np_file=f'/local/scratch-2/asv34/graph_weather/dataset/2022_{month}_normed.npy') for month in [1,4,7,10]]
datasets = [DataLoader(ds, batch_size=1, num_workers=32) for ds in ds_list]

filepaths = glob.glob("/local/scratch-2/asv34/graph_weather/dataset/2022/*")
coarsen = 8 # change this in preprocessor too if changed here
data = xr.open_zarr(filepaths[0], consolidated=True).coarsen(latitude=coarsen, boundary="pad").mean().coarsen(longitude=coarsen).mean()
lat_lons = np.array(np.meshgrid(data.latitude.values, data.longitude.values)).T.reshape(-1, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

criterion = NormalizedMSELoss(lat_lons=lat_lons, feature_variance=[1,1], device=device).to(device)
means = []


# train_dataset = dataset[:20]
# val_dataset = dataset[20:]
train_count = 95

# models = [GraphWeatherForecaster(lat_lons, feature_dim=42, num_blocks=6).to(device) for i in range(num_steps)]
# model = ParallelForecaster(lat_lons=lat_lons, models=models, num_steps=num_steps, feature_dim=42).to(device)
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
    val_loss = 0.0
    total_val_count = 0
    total_train_count = 0
    
    for j, dataset in enumerate(datasets):
        val_count = len(ds_list[j]) - train_count
        total_train_count += train_count
        total_val_count += val_count
        # start = time.time()
        # print(f"Start Epoch: {epoch+1}")
        for i, data in enumerate(dataset):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].float().to(device), data[1].float().to(device)
            if i < train_count: # use first 95 for training and the remaining for validation
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
                    val_loss += loss.item()
    
    print(f"train loss after epoch {epoch+1} is {running_loss/total_train_count}.")
    print(f"val loss after epoch {epoch+1} is {val_loss/total_val_count}.")

print("Finished Training")
torch.save(model.state_dict(), '/local/scratch-2/asv34/graph_weather/dataset/models/2022_4months_normed_singlemodel_100epochs.pt')