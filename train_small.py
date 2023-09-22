#from __future__ import absolute_import
from graph_weather import AnalysisDataset, GraphWeatherForecaster, ParallelDataset, ParallelForecaster, MultiResoDataset, MultiResoForecaster
import glob
import numpy as np
import xarray as xr
import torch
from torch.utils.data import DataLoader
from graph_weather.models.losses import NormalizedMSELoss
import torch.optim as optim
import sys
import matplotlib.pyplot as plt
import time

num_steps = 3
num_blocks = 6
feature_dim = 42
model_type = sys.argv[1]
cuda_num = sys.argv[2]
lr = 10 ** (-int(sys.argv[3]))
train_count = 95
num_epochs = 100
months = [1, 4, 7, 10]
# regional = True

device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
print(device)


filepaths = [glob.glob(f"/local/scratch-2/asv34/graph_weather/dataset/final/{region}/{region}_2022/*") for region in ['global', 'europe', 'uk']]
coarsen = 8 # change this in preprocessor too if changed here
uk_coarsen = 8
datas = [xr.open_zarr(filepath[0], consolidated=True).coarsen(latitude=coarsen, boundary="pad").mean().coarsen(longitude=coarsen, boundary="pad").mean() for filepath in filepaths]
lat_lons = [np.array(np.meshgrid(data.latitude.values, data.longitude.values)).T.reshape(-1, 2) for data in datas]


if model_type == 'global':
    ds_list = [AnalysisDataset(np_file=f'/local/scratch-2/asv34/graph_weather/dataset/final/global/normed_w_uk_at_coarsen{uk_coarsen}/global_2022_coarsen{coarsen}_{str(month).zfill(2)}_normed.npy') for month in months]
    model = GraphWeatherForecaster(lat_lons=lat_lons[0], regional=False,  feature_dim=feature_dim, num_blocks=num_blocks).to(device)
    criterion = NormalizedMSELoss(lat_lons=lat_lons[0], feature_variance=[1,1], device=device).to(device)

elif model_type == 'vector' or model_type == 'diff_pool':
    ds_list = [
        MultiResoDataset(
            np_files = [
                f'/local/scratch-2/asv34/graph_weather/dataset/final/{region}/normed_w_uk_at_coarsen{uk_coarsen}/{region}_2022_coarsen{coarsen}_{str(month).zfill(2)}_normed.npy' for region in ['global', 'europe', 'uk']
                ],
            global_gap=1,
            europe_gap=1,
            uk_gap=1
                         )
        for month in months
    ]
    
    model = MultiResoForecaster(lat_lons_list=lat_lons, resolutions=[2,3,4], feature_dim=feature_dim, num_blocks=num_blocks, attention_type=model_type)
    criterion = NormalizedMSELoss(lat_lons=lat_lons[2], feature_variance=[1,1], device=device).to(device)

elif model_type == 'uk':
    ds_list = [AnalysisDataset(np_file=f'/local/scratch-2/asv34/graph_weather/dataset/final/uk/normed_w_uk_at_coarsen{uk_coarsen}/uk_2022_coarsen{coarsen}_{str(month).zfill(2)}_normed.npy') for month in months]
    model = GraphWeatherForecaster(lat_lons=lat_lons[2], regional=True,  feature_dim=feature_dim, num_blocks=num_blocks).to(device)
    criterion = NormalizedMSELoss(lat_lons=lat_lons[2], feature_variance=[1,1], device=device).to(device)
else:
    exit()



# if model_type == 'single':
#     ds_list = [AnalysisDataset(np_file=f'/local/scratch-2/asv34/graph_weather/dataset/uk_2022_{month}_normed.npy') for month in months]
#     model = GraphWeatherForecaster(lat_lons=lat_lons, regional=regional,  feature_dim=feature_dim, num_blocks=num_blocks).to(device)
# else:
#     ds_list = [ParallelDataset(np_file=f'/local/scratch-2/asv34/graph_weather/dataset/uk_2022_{month}_normed.npy', num_steps=num_steps) for month in months]
#     model = ParallelForecaster(lat_lons=lat_lons, regional=regional, num_steps=num_steps, feature_dim=feature_dim, model_type=model_type, num_blocks=num_blocks).to(device)

datasets = [DataLoader(ds, batch_size=1, num_workers=32) for ds in ds_list]
optimizer = optim.AdamW(model.parameters(), lr=lr)


def plot_graph(num_epochs, train_losses, val_losses, model_type, lr):
    epochs = range(1,num_epochs + 1)
    plt.rcParams['figure.figsize'] = [12, 5]
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(f'{model_type} at {lr=}')
    ax1.set_title('Training loss vs epoch')
    ax2.set_title('Validation loss vs epoch')
    ax1.plot(epochs, train_losses)
    ax2.plot(epochs, val_losses)
    # plt.show()
    fig.savefig(f'/local/scratch-2/asv34/graph_weather/plots/final/plot_{model_type}_lr{lr}_blocks{num_blocks}_{num_epochs}epochs.png')


param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print('model size: {:.3f}MB'.format(size_all_mb))

print("Done Setup")


train_losses = []
val_losses = []
for epoch in range(num_epochs):  # loop over the dataset multiple times
    start_time = time.time()
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
    end_time = time.time()
    print(f"train loss after epoch {epoch+1} is {running_loss/total_train_count}.")
    print(f"val loss after epoch {epoch+1} is {val_loss/total_val_count}.")
    print(f'epoch {epoch+1} took {end_time-start_time} seconds')

print(f"Finished Training at {lr=}")
print(f'train_losses:\n{train_losses}\n')
print(f'val_losses:\n{val_losses}\n\n')
plot_graph(num_epochs=num_epochs, train_losses=train_losses, val_losses=val_losses, model_type=model_type, lr=lr)
torch.save(model.state_dict(), f'/local/scratch-2/asv34/graph_weather/dataset/models/final/{model_type}_lr{lr}_blocks{num_blocks}_{num_epochs}epochs.pt')
