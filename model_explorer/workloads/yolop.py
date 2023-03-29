import torch


model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)

model.gr = 1.0
model.nc = 1
