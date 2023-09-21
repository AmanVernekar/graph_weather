import glob
import xarray as xr
import numpy as np

# import pandas as pd
# from pysolar.util import extraterrestrial_irrad
# from . import const

coarsen = 1 # change this in train_small too if changed here
train = True # set to true if normalising UK training data, set to false otherwise
region = 'uk' # choose from 'uk', 'europe', or 'global'

datasets = []
months = ['01', '04', '07', '10']

for month in months:
    filepaths = glob.glob(f"/local/scratch-2/asv34/graph_weather/dataset/final/{region}/{region}_2022/2022_{month}*")
    filepaths = sorted(filepaths)

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
    datasets.append(dataset)

if train and region == 'uk':
    datasets_np = np.concatenate(datasets, axis=0)

    means = np.mean(datasets_np, axis=0)
    means = np.mean(means, axis=0)

    variances = (datasets_np - means)**2
    variances = np.mean(variances, axis=0)
    variances = np.mean(variances, axis=0)
    stdevs = np.sqrt(variances)

    print(f'means are {list(means)}\n\n')
    print(f'variances are {list(variances)}\n\n')
    print(f'stdevs are {list(stdevs)}\n\n')

else:
    means = [2.80617824e-06, 5.43949892e-06, 5.89384711e-05, 4.05640756e-04,
            1.57212126e-03, 4.61412586e-03, 6.97914594e-03, 2.11679876e+02,
            2.13045460e+02, 2.23016434e+02, 2.42570088e+02, 2.61344528e+02,
            2.74745165e+02, 2.81244825e+02, 5.25101776e+00, 1.32614835e+01,
            1.31099705e+01, 8.77430762e+00, 4.81280448e+00, 1.41418756e+00,
            -2.96233161e-02, 3.80625155e-03, 1.26068248e-03, -4.46992568e-02,
            -2.64150941e-02, -3.06685161e-02, 1.38253774e-01, 1.94378445e-01,
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

    stdevs  = [4.18465428e-07, 3.83990121e-06, 7.49772778e-05, 5.33798511e-04, 
            1.76814883e-03, 4.05898126e-03, 5.81521475e-03, 1.05035009e+01, 
            8.95322933e+00, 8.77334063e+00, 1.26320959e+01, 1.35027249e+01,
            1.55999942e+01, 1.70465191e+01, 1.66133106e+01, 1.61638726e+01,
            1.76731631e+01, 1.39142722e+01, 1.01917692e+01, 8.13043488e+00,
            6.05161854e+00, 6.78361625e+00, 9.72544498e+00, 1.34094115e+01,
            1.11865339e+01, 7.79281472e+00, 6.15354602e+00, 5.14933794e+00,
            9.87786063e-03, 3.56042137e-02, 8.43679505e-02, 1.44908923e-01,
            1.63292498e-01, 1.51776531e-01, 1.04199779e-01, 6.43888439e+03,
            6.03232724e+03, 5.63486953e+03, 4.24285376e+03, 2.79031604e+03,
            1.55183194e+03, 1.13716400e+03]

for i, dataset in enumerate(datasets):
    new = (dataset - means)/stdevs
    month = months[i]
    np.save(f'/local/scratch-2/asv34/graph_weather/dataset/final/{region}/{region}_2022_{month}_normed.npy', new)






   
    
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