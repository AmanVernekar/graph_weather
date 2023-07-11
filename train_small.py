#from __future__ import absolute_import
from graph_weather import AnalysisDataset, GraphWeatherForecaster
import glob

filepaths = glob.glob("/local/scratch-2/asv34/graph_weather/dataset/one_day/*")

ds = AnalysisDataset(filepaths, '/local/scratch-2/asv34/graph_weather/ls_mask.zarr', 0, 0, 1)

print(ds[0])
print(ds[0].shape)

print(ds[5])
print(ds[5].shape)

print(ds[22])
print(ds[22].shape)