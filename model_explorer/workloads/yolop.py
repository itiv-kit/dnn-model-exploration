import torch

from model_explorer.accuracy_functions.yolop_combined_accuracy import compute_yolop_combined_metric


def yolop_bdd100k_accuracy(base_model, dataloader_generator, progress=True, title=""):
    return compute_yolop_combined_metric(base_model, dataloader_generator)


model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)

model.gr = 1.0
model.nc = 1

accuracy_function = yolop_bdd100k_accuracy
