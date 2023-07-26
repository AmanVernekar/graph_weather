import torch
from graph_weather import AnalysisDataset, GraphWeatherForecaster, ParallelDataset, ParallelForecaster
import numpy as np
import sys
import glob
import xarray as xr
from torch.utils.data import DataLoader

model_file = sys.argv[1]
model_type = sys.argv[2]
num_blocks = int(sys.argv[3])
cuda_num = sys.argv[4]
# months = [3,6,9,12]
months = [1,4,7,10]
feature_dim = 42
num_steps = 3

device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
print(device)

means = [2.80617824e-06, 5.43949892e-06, 5.89384711e-05, 4.05640756e-04,
         1.57212126e-03, 4.61412586e-03, 6.97914594e-03, 2.11679876e+02,
         2.13045460e+02, 2.23016434e+02, 2.42570088e+02, 2.61344528e+02,
         2.74745165e+02, 2.81244825e+02, 5.25101776e+00, 1.32614835e+01,
         1.31099705e+01, 8.77430762e+00, 4.81280448e+00, 1.41418756e+00
         -2.96233161e-02, 3.80625155e-03, 1.26068248e-03 -4.46992568e-02
         -2.64150941e-02 -3.06685161e-02, 1.38253774e-01, 1.94378445e-01
         -1.60968473e-05, 8.82189535e-05, 1.05546063e-04, 4.59542703e-04,
         3.92029660e-05, 1.17745746e-02, 2.03996887e-02, 1.99296662e+05,
         1.33222749e+05, 1.01314354e+05, 7.00131214e+04, 4.06453642e+04,
         1.37232086e+04, 6.97392413e+02]

variances = [1.75113315e-13, 1.47448413e-11, 5.62159218e-09, 2.84940850e-07,
             3.12635029e-06, 1.64753289e-05, 3.38167226e-05, 1.10323532e+02,
             8.01603154e+01, 7.69715058e+01, 1.59569848e+02, 1.82323579e+02,
             2.43359820e+02, 2.90583813e+02, 2.76002090e+02, 2.61270779e+02,
             3.12340693e+02, 1.93606972e+02, 1.03872159e+02, 6.61039713e+01,
             3.66220870e+01, 4.60174495e+01, 9.45842801e+01, 1.79812318e+02,    
             1.25138541e+02, 6.07279613e+01, 3.78661286e+01, 2.65156812e+01,
             9.75721307e-05, 1.26766003e-03, 7.11795108e-03, 2.09985961e-02,
             2.66644398e-02, 2.30361155e-02, 1.08575939e-02, 4.14592322e+07,
             3.63889719e+07, 3.17517546e+07, 1.80018081e+07, 7.78586359e+06,
             2.40818236e+06, 1.29314196e+06]

stdevs  = torch.tensor([4.18465428e-07, 3.83990121e-06, 7.49772778e-05, 5.33798511e-04, 
           1.76814883e-03, 4.05898126e-03, 5.81521475e-03, 1.05035009e+01, 
           8.95322933e+00, 8.77334063e+00, 1.26320959e+01, 1.35027249e+01,
           1.55999942e+01, 1.70465191e+01, 1.66133106e+01, 1.61638726e+01,
           1.76731631e+01, 1.39142722e+01, 1.01917692e+01, 8.13043488e+00,
           6.05161854e+00, 6.78361625e+00, 9.72544498e+00, 1.34094115e+01,
           1.11865339e+01, 7.79281472e+00, 6.15354602e+00, 5.14933794e+00,
           9.87786063e-03, 3.56042137e-02, 8.43679505e-02, 1.44908923e-01,
           1.63292498e-01, 1.51776531e-01, 1.04199779e-01, 6.43888439e+03,
           6.03232724e+03, 5.63486953e+03, 4.24285376e+03, 2.79031604e+03,
           1.55183194e+03, 1.13716400e+03]).to(device)


filepaths = glob.glob("/local/scratch-2/asv34/graph_weather/dataset/2022/*")
coarsen = 8 # change this in preprocessor too if changed here
data = xr.open_zarr(filepaths[0], consolidated=True).coarsen(latitude=coarsen, boundary="pad").mean().coarsen(longitude=coarsen).mean()
lat_lons = np.array(np.meshgrid(data.latitude.values, data.longitude.values)).T.reshape(-1, 2)

if model_type == 'single':
    ds_list = [AnalysisDataset(np_file=f'/local/scratch-2/asv34/graph_weather/dataset/2022_{month}_normed.npy') for month in months]
    model = GraphWeatherForecaster(lat_lons=lat_lons, feature_dim=feature_dim, num_blocks=num_blocks).to(device)
else:
    ds_list = [ParallelDataset(np_file=f'/local/scratch-2/asv34/graph_weather/dataset/2022_{month}_normed.npy', num_steps=num_steps) for month in months]
    model = ParallelForecaster(lat_lons=lat_lons, num_steps=num_steps, feature_dim=feature_dim, model_type=model_type, num_blocks=num_blocks).to(device)

datasets = [DataLoader(ds, batch_size=1, num_workers=32) for ds in ds_list]

model_file = f'/local/scratch-2/asv34/graph_weather/dataset/models/{model_file}'
model.load_state_dict(torch.load(model_file))
model.eval()


n = 0
for ds in ds_list:
    n += len(ds)
se_sum = torch.zeros((16380,42))


index = 0
for j, dataset in enumerate(datasets):
    for i, data in enumerate(dataset):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].float().to(device), data[1].float().to(device)
        with torch.no_grad():
            outputs = model(inputs)
            se = ((torch.mul(stdevs, outputs - labels)) ** 2)[0]
            se_sum = se_sum + se

mse = se_sum/n
mse = np.mean(mse, axis=0)
rmse = mse ** 0.5

print('mse is:\n')
print(mse)
print('rmse is:\n')
print(rmse)






# dataset = []
# outputs = []

# pred_vals = outputs * stdevs + means
# true_vals = dataset * stdevs + means


# mse = np.mean(se, axis=0)
# mse = np.mean(mse, axis=0)
# rmse = mse ** 0.5






# exit()
# # model_dict['']

# print("Model's state_dict:")
# for param_tensor in model_dict:
#     print(param_tensor, "\t", model_dict[param_tensor].size())


# print("\n\n\n")

# filepath2 = '/local/scratch-2/asv34/graph_weather/dataset/models/2022_4months_normed_simple_attention_lr0.0001_100epochs.pt'
# model_dict = torch.load(filepath2)
# # model_dict['']

# print("Model's state_dict:")
# for param_tensor in model_dict:
#     print(param_tensor, "\t", model_dict[param_tensor].size())

# exit()

