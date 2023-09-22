import glob
import xarray as xr
import numpy as np
import sys

# import pandas as pd
# from pysolar.util import extraterrestrial_irrad
# from . import const

coarsen = int(sys.argv[1]) # change this in train_small too if changed here
train = True if sys.argv[2] == 'train' else False # set to true if normalising UK training data, set to false otherwise
region = sys.argv[3] # choose from 'uk', 'europe', or 'global'
uk_coarsen = int(sys.argv[4])

datasets = []
train_months = ['01', '04', '07', '10']
test_months = ['03', '06', '09', '12']
months = train_months if train else test_months
months = train_months.extend(test_months)

if not (train and region == 'uk'):
    skip = True
    
    if uk_coarsen == 1:
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
    
    elif uk_coarsen == 8:
        means = [3.017617372178291e-06, 4.026955072268279e-06, 3.919694996054057e-05, 0.0003424855627245857, 0.0012913728432022127, 0.0038623022135732005, 
                 0.006368997608006738, 215.26357722906994, 217.48205627615428, 219.74546409363995, 241.39648957368797, 262.09996101505175, 276.76512048733366, 
                 283.755363690067, 8.520655246650152, 14.054017511670452, 16.786359494240422, 13.13613145454615, 9.389745695105594, 5.632234733034023, 
                 2.4006661617118437, -2.8882441575848334, -1.8282390766130188, -0.9088271104344927, 0.09732278016159893, 0.6668588425053402, 1.1378962739411562, 
                 1.0291849100179764, 0.00023861044792448114, 0.00028923511690138044, -0.002952227744588588, -0.00381537134694328, -0.0029659489557100783, 
                 -0.000572505279416895, 0.0019517150563906824, 201711.76085863396, 133623.3754461576, 101743.01007835436, 70753.63182910068, 41415.64177810683, 
                 14400.277316347587, 1299.468715709076]

        variances = [1.0078323610226312e-14, 2.2649646600510817e-12, 9.557947675975751e-10, 7.20349234180751e-08, 9.367066666747196e-07, 3.3014757743941813e-06, 
                     4.197026644549442e-06, 38.23032291535325, 29.77210913814644, 21.434382505666147, 32.53809530930581, 31.987021174507262, 34.544350815219126, 
                     27.08174847901494, 205.82473531644354, 115.03983134189882, 269.57073382978604, 178.32578522047686, 95.45161382826637, 65.08365142536047, 
                     38.412888071888084, 67.70937004423419, 131.00880823371918, 370.4436093825208, 237.68108353170504, 116.26539509829733, 64.16227681718507, 
                     38.13966260141843, 0.00022419431785689853, 0.0011582304740874057, 0.007398154628469788, 0.03322657295288776, 0.04445411676774483, 
                     0.03601297302043996, 0.002341214760106314, 11043501.005494595, 6348995.101647777, 6128890.71323172, 3848124.965546524, 1933765.1221418243, 
                     989325.9220014111, 905065.6972223829]

        stdevs = [1.0039085421604057e-07, 1.504979953371832e-06, 3.091593064420955e-05, 0.0002683932253580092, 0.0009678360742784491, 0.0018169963605891403, 
                  0.00204866460030661, 6.183067435775971, 5.456382422278193, 4.629728124379027, 5.704217326619474, 5.655706956208681, 5.87744424177883, 
                  5.204012728560043, 14.34659316062331, 10.725662279873388, 16.41860937563794, 13.353867800022465, 9.769934177273988, 8.067443921426444, 
                  6.197813168520659, 8.228570352390152, 11.445907925268278, 19.246911684281216, 15.416909013537865, 10.782643233377303, 8.010135880070017, 
                  6.175731746232056, 0.01497311984380338, 0.03403278528253904, 0.0860125259974952, 0.18228157601054407, 0.21084144935886023, 0.1897708434413463, 
                  0.04838610089794707, 3323.17634282242, 2519.721234908294, 2475.6596521395504, 1961.6638258240182, 1390.5988358048573, 994.64864248709, 
                  951.3494085888648]
else:
    skip = False

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
    if skip:
        new = (dataset - means)/stdevs
        np.save(f'/local/scratch-2/asv34/graph_weather/dataset/final/{region}/normed_w_uk_at_coarsen{uk_coarsen}/{region}_2022_coarsen{coarsen}_{month}_normed.npy', new)        
    else:
        datasets.append(dataset)



if not skip:
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

    for i, dataset in enumerate(datasets):
        new = (dataset - means)/stdevs
        month = months[i]
        np.save(f'/local/scratch-2/asv34/graph_weather/dataset/final/{region}/normed_w_uk_at_coarsen{uk_coarsen}/{region}_2022_coarsen{coarsen}_{month}_normed.npy', new)






   
    
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