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

from pytorch_quantization import quant_modules
quant_modules.initialize()

quant_desc_input = QuantDescriptor(calib_method='histogram')
quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

model = models.resnet50(pretrained=True)


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

transforms = transforms.Compose([transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize])

dataset = datasets.ImageFolder('/home/oq4116/temp/ILSVRC/Data/CLS-LOC/val', transforms)
indices = random.sample(range(len(dataset)), 100)
dataset_100 = Subset(dataset, indices=indices)
indices = random.sample(range(len(dataset)), 1000)
dataset_1000 = Subset(dataset, indices=indices)


def collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistic"""

    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
        model(image)
        if i >= num_batches:
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

dataloader = DataLoader(dataset_1000, batch_size=64, shuffle=True, pin_memory=True)
# It is a bit slow since we collect histograms on CPU
with torch.no_grad():
    collect_stats(model, dataloader, num_batches=4)
    compute_amax(model, method="percentile", percentile=99.99)


dataloader = DataLoader(dataset_100, batch_size=64, shuffle=True, pin_memory=True)
correct_pred = 0
with torch.no_grad():
    for X, y_true in tqdm(dataloader):
        y_prob = model(X)
        _, predicted_labels = torch.max(y_prob, 1)
        print(y_true)

        correct_pred += (predicted_labels == y_true).sum()

accuracy = correct_pred.float() / len(dataloader.dataset)
print(accuracy)

quant_modules.deactivate()
correct_pred = 0
with torch.no_grad():
    for X, y_true in tqdm(dataloader):
        y_prob = model(X)
        _, predicted_labels = torch.max(y_prob, 1)
        print(y_true)

        correct_pred += (predicted_labels == y_true).sum()

accuracy = correct_pred.float() / len(dataloader.dataset)
print(accuracy)



