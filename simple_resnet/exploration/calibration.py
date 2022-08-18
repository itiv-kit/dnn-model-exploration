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

quant_desc_input = QuantDescriptor(calib_method='histogram')
quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

dev_string = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(dev_string)

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model = model.to(device)


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

transforms = transforms.Compose([transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize])

dataset = datasets.ImageFolder('/home/oq4116/temp/ILSVRC/Data/CLS-LOC/val', transforms)


def collect_stats(model, data_loader):
    """Feed data to the network and collect statistic"""

    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    for image, _ in tqdm(data_loader):
        model(image.to(device))

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

dataloader = DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True)
# It is a bit slow since we collect histograms on CPU
with torch.no_grad():
    collect_stats(model, dataloader)
    compute_amax(model, method="percentile", percentile=99.99)

torch.save(model.state_dict(), "resnet50-calib.pth")


print(summary(model, (32, 3, 224, 224)))


