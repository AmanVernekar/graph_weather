import torch

filepath = '/local/scratch-2/asv34/graph_weather/dataset/models/2022_4months_normed_single_lr4_100epochs.pt'
model_dict = torch.load(filepath)
model_dict['']

print("Model's state_dict:")
for param_tensor in model_dict:
    print(param_tensor, "\t", model_dict[param_tensor].size())