import torch
from torchvision import models
from commons import dataset, data_sampler, device
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import io
import sys
import pickle

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
quant_modules.initialize()


model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.load_state_dict(torch.load('resnet50-calib.pth', map_location=device))

histogram_data = {}
BATCHES = 1

dataloader = DataLoader(dataset, batch_size=32, sampler=data_sampler)

# clean up
dir = 'images'
for f in os.listdir(dir):
    os.remove(os.path.join(dir, f))
 
# get bit combination...
from explore import LayerwiseQuantizationProblem, QuantizationModel

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

with open('exploration_20_25.pkl', 'rb') as f:
    d = CPU_Unpickler(f).load()

bits = d.opt[0].get("X")


# 0. prepare model
def collection_hook(module, inp, outp, module_name):
    if module_name in histogram_data:
        histogram_data[module_name] = torch.concat([histogram_data[module_name], outp.flatten()])
    else:
        histogram_data[module_name] = outp.flatten()

handles = []
layernames = []

for name, module in model.named_modules():
    if isinstance(module, quant_nn.TensorQuantizer):
        module.disable()
        layernames.append(name)
        handle = module.register_forward_hook(
            lambda module, inp, out, module_name=name:
            collection_hook(module, inp, out, module_name)
        )
        handles.append(handle)
    
# 1. run model without quantization
with torch.no_grad():
    for i, (X, _) in tqdm(enumerate(dataloader), desc="Collecting Non-Quant"):
        model(X)
        if i >= BATCHES:
            break
    
# 2. run model with quantization
histogram_data_no_quant = histogram_data.copy()
histogram_data.clear()

bit_widths = {}
for i, name in enumerate(layernames):
    bit_widths[name] = int(bits[i])

for name, module in model.named_modules():
    if isinstance(module, quant_nn.TensorQuantizer):
        module.enable()
        module.enable_quant()
        module.num_bits = bit_widths[name]


with torch.no_grad():
    for i, (X, _) in tqdm(enumerate(dataloader), desc="Collecting Quant"):
        model(X)
        if i >= BATCHES:
            break


for handle in handles:
    handle.remove()

# 3. plot diagrams ... 
for i, k in tqdm(enumerate(histogram_data.keys()), desc="Rendering"):
    data_no_quant = histogram_data_no_quant[k].numpy()
    data_quant = histogram_data[k].numpy()
    path = os.path.join('images', '{}.png'.format(i))

    plt.subplot(2, 1, 1)
    plt.title("Layer: {}, Bits: {}".format(k, bit_widths[k]))
    plt.hist(data_no_quant, bins=256)

    plt.subplot(2, 1, 2)
    plt.hist(data_quant, bins=256)

    plt.savefig(path)
    plt.clf()

