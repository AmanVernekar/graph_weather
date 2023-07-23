#from __future__ import absolute_import
from graph_weather import AnalysisDataset, GraphWeatherForecaster, ParallelDataset, ParallelForecaster
import glob
import numpy as np
import xarray as xr
import torch
from torch.utils.data import DataLoader
from graph_weather.models.losses import NormalizedMSELoss
import torch.optim as optim
import sys
import matplotlib.pyplot as plt

num_steps = 3
model_type = sys.argv[0]
cuda_num = sys.argv[1]
train_count = 95
num_epochs = 100


n = 5
start = 1e-6
ratio = 100**0.25
learning_rates = [start * ratio**i for i in range(n)]


filepaths = glob.glob("/local/scratch-2/asv34/graph_weather/dataset/2022/*")
coarsen = 8 # change this in preprocessor too if changed here
data = xr.open_zarr(filepaths[0], consolidated=True).coarsen(latitude=coarsen, boundary="pad").mean().coarsen(longitude=coarsen).mean()
lat_lons = np.array(np.meshgrid(data.latitude.values, data.longitude.values)).T.reshape(-1, 2)


device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
print(device)


if model_type == 'single':
    ds_list = [AnalysisDataset(np_file=f'/local/scratch-2/asv34/graph_weather/dataset/2022_{month}_normed.npy') for month in [1,4,7,10]]
    model = GraphWeatherForecaster(lat_lons, feature_dim=42, num_blocks=6).to(device)
elif model_type == 'linear' or model_type == 'params':
    ds_list = [ParallelDataset(np_file=f'/local/scratch-2/asv34/graph_weather/dataset/2022_{month}_normed.npy', num_steps=num_steps) for month in [1,4,7,10]]
    model = ParallelForecaster(lat_lons=lat_lons, num_steps=num_steps, feature_dim=42, model_type=model_type).to(device)


datasets = [DataLoader(ds, batch_size=1, num_workers=32) for ds in ds_list]
criterion = NormalizedMSELoss(lat_lons=lat_lons, feature_variance=[1,1], device=device).to(device)


param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print('model size: {:.3f}MB'.format(size_all_mb))

print("Done Setup")


def plot_graph(num_epochs, train_losses, val_losses, model_type, lr, k):
    epochs = range(1,num_epochs + 1)
    plt.rcParams['figure.figsize'] = [12, 5]
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(f'{model_type} at {lr=}')
    ax1.set_title('Training loss vs epoch')
    ax2.set_title('Validation loss vs epoch')
    ax1.plot(epochs, train_losses)
    ax2.plot(epochs, val_losses)
    # plt.show()
    fig.savefig(f'/local/scratch-2/asv34/graph_weather/plots/plot_2022_4months_normed_{model_type}_lr{k}_{num_epochs}epochs.png')


for k, lr in enumerate(learning_rates):
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        val_loss = 0.0
        total_val_count = 0
        total_train_count = 0
        
        for j, dataset in enumerate(datasets):
            val_count = len(ds_list[j]) - train_count
            total_train_count += train_count
            total_val_count += val_count
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
                else:
                    model.eval()
                    with torch.no_grad():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
        
        train_losses.append(running_loss/total_train_count)
        val_losses.append(val_loss/total_val_count)
        print(f"train loss after epoch {epoch+1} is {running_loss/total_train_count}.")
        print(f"val loss after epoch {epoch+1} is {val_loss/total_val_count}.")

    print(f"Finished Training at {lr=}")
    print(f'train_losses:\n{train_losses}\n')
    print(f'val_losses:\n{val_losses}\n\n')
    plot_graph(num_epochs=num_epochs, train_losses=train_losses, val_losses=val_losses, model_type=model_type, lr=lr, k=k)
    torch.save(model.state_dict(), f'/local/scratch-2/asv34/graph_weather/dataset/models/2022_4months_normed_{model_type}_lr{k}_{num_epochs}epochs.pt')
