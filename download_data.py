import cdsapi
import xarray as xr
import zarr
import numcodecs

c = cdsapi.Client()
uk_area = [63, -10, 47, 4]
europe_area = [82, -38, 28, 32]

for year in [2022]:
    for month in [1, 4, 7, 10]:
        for day in range(1,32):
            for time in ['00:00', '06:00', '12:00', '18:00']:
                try:
                    c.retrieve(
                        'reanalysis-era5-pressure-levels',
                        {
                            'product_type': 'reanalysis',
                            'variable': [
                                'geopotential', 'specific_humidity','temperature',
                                'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity',
                            ],
                            'pressure_level': [
                                '50', '150', '250', '400', '600', '850', '1000'
                            ],
                            'year': str(year),
                            'month': str(month).zfill(2),
                            'day': str(day).zfill(2),
                            'time': time,
                            'area': europe_area,
                            'format': 'netcdf',
                        },
                        f'/local/scratch-2/asv34/graph_weather/dataset/europe_2022/download_air.nc')
                    data = xr.open_dataset("/local/scratch-2/asv34/graph_weather/dataset/europe_2022/download_air.nc", engine="netcdf4")
                    #print(data)
                    encoding = {var: {"compressor": numcodecs.get_codec(dict(id="zlib", level=5))} for var in data.data_vars}
                    d = data.chunk({"time": 1})
                    with zarr.ZipStore(f'/local/scratch-2/asv34/graph_weather/dataset/europe_2022/{year}_{str(month).zfill(2)}_{str(day).zfill(2)}_{time[:2]}.zarr.zip', mode='w') as store:
                        d.to_zarr(store, encoding=encoding, compute=True)
                except:
                    continue