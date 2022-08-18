import sys
import torch
import torch.utils.data
from torch import nn
import random
from tqdm import tqdm

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor

from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Subset

from torchinfo import summary

from pytorch_quantization import quant_modules
quant_modules.initialize()


#load model:
torch.load('resnet50-calib.pth')


dataloader = DataLoader(dataset_10000, batch_size=64, shuffle=True, pin_memory=True)
correct_pred = 0
model = model.to(device)
with torch.no_grad():
    for X, y_true in tqdm(dataloader):
        X = X.to(device)
        y_true = y_true.to(device)
        y_prob = model(X)
        _, predicted_labels = torch.max(y_prob, 1)

        correct_pred += (predicted_labels == y_true).sum()

accuracy = correct_pred.float() / len(dataloader.dataset)
print(accuracy)

quant_modules.deactivate()
correct_pred = 0
model = model.to(device)
with torch.no_grad():
    for X, y_true in tqdm(dataloader):
        X = X.to(device)
        y_true = y_true.to(device)
        y_prob = model(X)
        _, predicted_labels = torch.max(y_prob, 1)

        correct_pred += (predicted_labels == y_true).sum()

accuracy = correct_pred.float() / len(dataloader.dataset)
print(accuracy)