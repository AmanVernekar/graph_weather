import glob
import torch

filepaths = glob.glob("/local/scratch-2/asv34/graph_weather/dataset/models/*")

for fp in filepaths:
    model_dict = torch.load(fp)
    # model_dict['']

    print("Model's state_dict:")
    for param_tensor in model_dict:
        print(param_tensor, "\t", model_dict[param_tensor].size())