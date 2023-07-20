"""

The dataloader has to do a few things for the model to work correctly

1. Load the land-0sea mask, orography dataset, regridded from 0.1 to the correct resolution
2. Calculate the top-of-atmosphere solar radiation for each location at fcurrent time and 10 other
 times +- 12 hours
3. Add day-of-year, sin(lat), cos(lat), sin(lon), cos(lon) as well
3. Batch data as either in geometric batches, or more normally
4. Rescale between 0 and 1, but don't normalize

"""

import numpy as np
from torch.utils.data import Dataset
import torch


class AnalysisDataset(Dataset):
    def __init__(self, np_file):  # filepaths, invariant_path, mean, std, coarsen: int = 8
        super().__init__()
        # self.filepaths = sorted(filepaths)
        # self.invariant_path = invariant_path
        # self.coarsen = coarsen
        # self.mean = mean
        # self.std = std
        self.dataset = torch.from_numpy(np.load(np_file))

    def __len__(self):
        return self.dataset.shape[0] - 1

    def __getitem__(self, item):
        return self.dataset[item], self.dataset[item + 1]



class ParallelDataset(Dataset):
    def __init__(self, np_file, num_steps):  # filepaths, invariant_path, mean, std, coarsen: int = 8
        super().__init__()
        self.dataset = torch.from_numpy(np.load(np_file))
        self.num_steps = num_steps

    def __len__(self):
        return self.dataset.shape[0] - self.num_steps

    def __getitem__(self, item):
        features = []
        for i in range(item, item + self.num_steps):
            features.append(self.dataset[i])
        return features, self.dataset[item + self.num_steps]




# obs_data = xr.open_zarr(
#     "/home/jacob/Development/prepbufr.gdas.20160101.t00z.nr.48h.raw.zarr", consolidated=True
# )
# # TODO Embedding? These should stay consistent across all of the inputs, so can just load the values, not the strings?
# # Should only take in the quality markers, observations, reported observation time relative to start point
# # Observation errors, and background values, lat/lon/height/speed of observing thing
# print(obs_data)
# print(obs_data.hdr_inst_typ.values)
# print(obs_data.hdr_irpt_typ.values)
# print(obs_data.obs_qty_table.values)
# print(obs_data.hdr_prpt_typ.values)
# print(obs_data.hdr_sid_table.values)
# print(obs_data.hdr_typ_table.values)
# print(obs_data.obs_desc.values)
# print(obs_data.data_vars.keys())
# exit()
# analysis_data = xr.open_zarr(
#     "/home/jacob/Development/gdas1.fnl0p25.2016010100.f00.zarr", consolidated=True
# )
# print(analysis_data)
