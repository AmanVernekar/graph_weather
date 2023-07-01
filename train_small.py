#from __future__ import absolute_import
from graph_weather import AnalysisDataset

ds = AnalysisDataset(['/local/scratch-2/asv34/graph_weather/out.zarr'], '/local/scratch-2/asv34/graph_weather/ls_mask.zarr', 0, 0, 1)
print(ds[0])
print(ds[0].shape)