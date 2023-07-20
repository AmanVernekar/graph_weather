import glob
import xarray as xr
import numpy as np

# import pandas as pd
# from pysolar.util import extraterrestrial_irrad
# from . import const

filepaths = glob.glob("/local/scratch-2/asv34/graph_weather/dataset/2022/*")
filepaths = sorted(filepaths)

coarsen = 8 # change this in train_small too if changed here

for i, f in enumerate(filepaths):
    if coarsen <= 1:  # Don't coarsen, so don't even call it
        zarr = xr.open_zarr(f, consolidated=True)
    else:
        zarr = (
            xr.open_zarr(f, consolidated=True)
            .coarsen(latitude=coarsen, boundary="pad")
            .mean()
            .coarsen(longitude=coarsen)
            .mean()
        )
    # data = np.stack(
    #     [
    #         zarr[f"{var}"].values
    #         for var in zarr.data_vars
    #     ],
    #     axis=-1,
    # )
    # data = data.T.reshape((-1, data.shape[-1]))

    data = np.concatenate(
        [zarr[var].values[0] for var in zarr.data_vars]
    ).T
    data = np.reshape(data, (-1, data.shape[-1]))

    if i == 0:
        dataset = np.ndarray(shape=[len(filepaths)] + list(data.shape), dtype=float)
    dataset[i] = data

means = np.mean(dataset, axis=0)
means = np.mean(means, axis=0)
dataset = dataset/means
np.save('/local/scratch-2/asv34/graph_weather/dataset/jan_2022_rescaled.npy', dataset)




   
    
    # lat_lons = np.array(np.meshgrid(zarr.latitude.values, zarr.longitude.values)).T.reshape(
    #     (-1, 2)
    # )
    # sin_lat_lons = np.sin(lat_lons)
    # cos_lat_lons = np.cos(lat_lons)
    # date = start.time.dt.date
    # day_of_year = start.time.dt.dayofyear / 365.0
    # np.sin(day_of_year)
    # np.cos(day_of_year)
    
    # Land-sea mask data, resampled to the same as the physical variables
    # landsea = (
    #     xr.open_zarr(invariant_path, consolidated=True)
    #     .interp(latitude=start.latitude.values)
    #     .interp(longitude=start.longitude.values)
    # )
    # # Calculate sin,cos, day of year, solar irradiance here before stacking
    # landsea = np.stack(
    #     [
    #         (landsea[f"{var}"].values - const.LANDSEA_MEAN[var]) / const.LANDSEA_STD[var]
    #         for var in landsea.data_vars
    #         if not np.isnan(landsea[f"{var}"].values).any()
    #     ],
    #     axis=-1,
    # )
    # landsea = landsea.T.reshape((-1, landsea.shape[-1]))


    # solar_times = [np.array([extraterrestrial_irrad(date, lat, lon) for lat, lon in lat_lons])]
    # for when in pd.date_range(
    #     date - pd.Timedelta("12 hours"), date + pd.Timedelta("12 hours"), freq="1H"
    # ):
    #     solar_times.append(
    #         np.array([extraterrestrial_irrad(when, lat, lon) for lat, lon in lat_lons])
    #     )
    # solar_times = np.array(solar_times)

    # End time solar radiation too
    # end_date = end.time.dt.date
    # end_solar_times = [
    #     np.array([extraterrestrial_irrad(end_date, lat, lon) for lat, lon in lat_lons])
    # ]
    # for when in pd.date_range(
    #     end_date - pd.Timedelta("12 hours"), end_date + pd.Timedelta("12 hours"), freq="1H"
    # ):
    #     end_solar_times.append(
    #         np.array([extraterrestrial_irrad(when, lat, lon) for lat, lon in lat_lons])
    #     )
    # end_solar_times = np.array(solar_times)

    # # Normalize to between -1 and 1
    # solar_times -= const.SOLAR_MEAN
    # solar_times /= const.SOLAR_STD
    # end_solar_times -= const.SOLAR_MEAN
    # end_solar_times /= const.SOLAR_STD

    # Stack the data into a large data cube
    
    # input_data = pd.DataFrame([start[f"{var}"].values for var in start.data_vars])
    # input_data = input_data.stack()

    # TODO Combine with above? And include sin/cos of day of year
    # print("hi")
    # print(input_data.shape)   
   
    # print(a.shape)
    # input_data = np.concatenate(
    #     [
    #         a,
    #         sin_lat_lons,
    #         cos_lat_lons
    #     ],
    #     axis=0,
    # ) # removed landsea and solar_times
    # # Not want to predict non-physics variables -> Output only the data variables? Would be simpler, and just add in the new ones each time
    # print("hi again")
   
    # output_data = np.concatenate(
    #     [
    #         output_data.T.reshape((-1, output_data.shape[-1])),
    #         sin_lat_lons,
    #         cos_lat_lons
    #     ],
    #     axis=-1,
    # ) # removed landsea and end_solar_times
    # Stick with Numpy, don't tensor it, as just going from 0 to 1

    # Normalize now

    #     output_data = np.stack(
    #     [
    #         end[f"{var}"].values
    #         for var in ['t', 'z', 'q', 'u', 'v', 'w']
    #     ],
    #     axis=-1,
    # )

    # output_data = output_data[0]
    # output_data = output_data.T.reshape((-1, output_data.shape[-1]))