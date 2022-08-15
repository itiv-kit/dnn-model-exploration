import pprint
import random

import torch
from torch.utils.data import DataLoader, Subset

from torchvision import models
from torchvision import transforms
from torchvision import datasets

from pytorch_quantization.tensor_quant import QuantDescriptor
import pytorch_quantization.nn as quant_nn

from tqdm import tqdm
import pandas as pd 


amaxes = {}

def conv2d_predicate(module):
    return isinstance(module, torch.nn.Conv2d)

def get_amax_hook(module, inp, outp, amax_name):
    amax = torch.amax(inp[0])
    prev_amax = amaxes[amax_name]
    if isinstance(prev_amax, float):
        prev_amax = torch.tensor(prev_amax)
    compare = torch.stack((amax, prev_amax))
    amaxes[amax_name] = torch.amax(compare)


m = models.resnet50(pretrained=True)

# prepare dataset
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

transforms = transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               normalize])

dataset = datasets.ImageFolder('/home/oq4116/temp/ILSVRC/Data/CLS-LOC/val/', transforms)
# indices = random.sample(range(len(dataset)), 100)
# dataset = Subset(dataset, indices=indices)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# modify model
for name, module in m.named_modules():
    if conv2d_predicate(module):
        amaxes[name] = 0.0

        module.register_forward_hook(
            lambda module, inp, out, amax_name=name: get_amax_hook(module, inp, out, amax_name)
        )

m.eval()
with torch.no_grad():
    for X, y_true in tqdm(dataloader):
        y_prob = m(X)

        
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(amaxes)

for k, v in amaxes.items():
    amaxes[k] = v.item()

df = pd.DataFrame(data=amaxes, index=[0])
df = (df.T)
df.to_excel('data.xlsx')

