import sys
import torch
import torch.utils.data
from torch import nn
import math
from tqdm import tqdm
from commons import get_dataloader, device

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor

from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Subset

from torchinfo import summary

from pytorch_quantization import quant_modules
quant_modules.initialize()

quant_desc_input = QuantDescriptor(calib_method='histogram')
quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model = model.to(device)

summary(model, input_size=(64, 3, 224, 224))


def collect_stats(model, data_loader, n_max=None, n_len=None):
    """Feed data to the network and collect statistic"""

    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    if n_max is None:
        for image, _ in tqdm(data_loader, total=n_len):
            model(image.to(device))
    else:
        for i, (image, _) in tqdm(enumerate(data_loader), total=n_max):
            model(image.to(device))
            if n_max is not None:
                if i >= n_max:
                    break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(F"{name:40}: {module}")
    model = model.to(device)

# It is a bit slow since we collect histograms on CPU
dataloader = get_dataloader(method='all', webdataset=True, batch_size=64, dataset='val')

with torch.no_grad():
    collect_stats(model, dataloader, n_max=None, n_len=math.ceil(50000/dataloader.batch_size))
    compute_amax(model, method="percentile", percentile=99.99)

torch.save(model.state_dict(), "resnet18-calib.pth")


# summary(model, (32, 3, 224, 224))


