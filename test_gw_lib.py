import torch
from graph_weather import GraphWeatherForecaster
from graph_weather.models.losses import NormalizedMSELoss

lat_lons = []
for lat in range(-90, 90, 1):
    for lon in range(0, 360, 1):
        lat_lons.append((lat, lon))
model = GraphWeatherForecaster(lat_lons)

features = torch.randn((2, len(lat_lons), 78))
print(features.shape)

out = model(features)
criterion = NormalizedMSELoss(lat_lons=lat_lons, feature_variance=torch.randn((78,)))
loss = criterion(out, features)
loss.backward()

print(out.shape)
print(loss)


# import torch
# from graph_weather import GraphWeatherAssimilator
# from graph_weather.models.losses import NormalizedMSELoss
# import numpy as np

# obs_lat_lons = []
# for lat in range(-90, 90, 7):
#     for lon in range(0, 180, 6):
#         obs_lat_lons.append((lat, lon, np.random.random(1)))
#     for lon in 360 * np.random.random(100):
#         obs_lat_lons.append((lat, lon, np.random.random(1)))

# output_lat_lons = []
# for lat in range(-90, 90, 5):
#     for lon in range(0, 360, 5):
#         output_lat_lons.append((lat, lon))
# model = GraphWeatherAssimilator(output_lat_lons=output_lat_lons, analysis_dim=24)

# features = torch.randn((1, len(obs_lat_lons), 2))
# lat_lon_heights = torch.tensor(obs_lat_lons)
# out = model(features, lat_lon_heights)
# assert not torch.isnan(out).all()
# assert out.size() == (1, len(output_lat_lons), 24)

# criterion = torch.nn.MSELoss()
# loss = criterion(out, torch.randn((1, len(output_lat_lons), 24)))
# loss.backward()
# print(loss)
# print(out.shape)