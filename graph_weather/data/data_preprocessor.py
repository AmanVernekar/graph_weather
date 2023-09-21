import glob
import xarray as xr
import numpy as np

# import pandas as pd
# from pysolar.util import extraterrestrial_irrad
# from . import const

coarsen = 4 # change this in train_small too if changed here
train = True # set to true if normalising UK training data, set to false otherwise
region = 'europe' # choose from 'uk', 'europe', or 'global'

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
                .coarsen(longitude=coarsen, boundary="pad")
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

else: # below values generated based on UK data for months 1, 4, 7 and 10 in 2022
    means = [3.0166762157579455e-06, 3.9560930879360215e-06, 3.8351314814273384e-05, 0.0003396595888137118, 0.0012867776079310803, 
             0.003833464493499311, 0.006286048431357801, 215.33928634354638, 217.76054517424703, 219.7321800302087, 241.1157702580787, 
             261.77968952403296, 276.3531956476422, 283.3917086280774, 8.895224371150183, 14.194233130727183, 17.267192941886787, 
             13.521757991065273, 9.715120118654424, 5.963451454067497, 2.61615314171837, -2.8521758652591993, -1.6414841911736555, 
             -0.5453749043898468, 0.38463832072662013, 0.8972209794270624, 1.3065404441973123, 1.2073039467007576, 0.0001799281565135242, 
             -8.623624376444664e-05, -0.00397386423372239, -0.0045973597422843265, -0.0035422798913821124, 0.000349529344915178, 0.0018045873695415, 
             201652.69816816857, 133495.76988445988, 101584.00499884949, 70620.3051726795, 41317.35425607574, 14337.780164844866, 1256.5864153205666]


    variances = [9.863395306163428e-15, 2.1378205680349545e-12, 9.88166433582512e-10, 7.90394918524078e-08, 1.0290635856244799e-06, 
                 3.4712386714278796e-06, 4.051078799531303e-06, 41.136430597322416, 31.241985994741093, 22.370688049475163, 32.15564373758235, 
                 30.99628212796478, 32.3362553545285, 23.749810613996402, 219.68072210100107, 118.60528389998366, 277.80617270829936, 187.02943825892203, 
                 99.55131524521134, 69.50994026433342, 42.47215574474621, 70.4142919215883, 131.51399030608906, 378.81673362790417, 246.6301229206252, 
                 120.86455834120771, 68.61862999485614, 41.24161530908041, 0.0004370338290810367, 0.0023204958228772925, 0.014021187423027468, 
                 0.06356108343078125, 0.08798190754653683, 0.08794815175565156, 0.01004232553005782, 11438210.68322133, 6206758.489265246, 
                 5988038.229831126, 3804535.9252497186, 1935952.0194407771, 1025892.1061642866, 951836.0778087376]


    stdevs = [9.931462785593785e-08, 1.4621287795659294e-06, 3.143511465833252e-05, 0.0002811396305261992, 0.0010144277133559, 0.0018631260481856507, 
              0.002012729191801844, 6.4137688294264565, 5.589453103367188, 4.72976617281184, 5.670594654670915, 5.567430478054017, 5.6864976351466545, 
              4.873377741771758, 14.821630210641509, 10.890605304572546, 16.66751849281408, 13.675870658167327, 9.977540540895404, 8.337262156387636, 
              6.5170664984137, 8.39132241792605, 11.467954931289583, 19.463214884183554, 15.704461879371264, 10.993841837192662, 8.283636278522623, 
              6.421963508856182, 0.020905354076911416, 0.04817152502129544, 0.11841109501658816, 0.25211323533440533, 0.2966174430921702, 0.29656053640977176, 
              0.10021140419162791, 3382.0423834158746, 2491.3366872555075, 2447.046838503735, 1950.521962257723, 1391.384928566059, 1012.863320574048, 
              975.6208678624795]

for i, dataset in enumerate(datasets):
    new = (dataset - means)/stdevs
    month = months[i]
    np.save(f'/local/scratch-2/asv34/graph_weather/dataset/final/{region}/{region}_2022_coarsen{coarsen}_{month}_normed.npy', new)






   
    
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