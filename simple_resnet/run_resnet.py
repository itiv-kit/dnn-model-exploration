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


dev_string = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(dev_string)
amaxes = {}

def conv2d_predicate(module):
    return isinstance(module, torch.nn.Conv2d)

def get_amax_hook(module, inp, outp, amax_name):
    cpu_dev = torch.device("cpu")

    amax = torch.amax(inp[0]).to(cpu_dev)
    prev_amax = amaxes[amax_name]

    compare = torch.stack((amax, prev_amax))
    amaxes[amax_name] = torch.amax(compare)


m = models.resnet50(pretrained=True)
m.to(device)

# prepare dataset
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

transforms = transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               normalize])

dataset = datasets.ImageFolder('/data/oq4116/imagenet/val', transforms)
# indices = random.sample(range(len(dataset)), 100)
# dataset = Subset(dataset, indices=indices)

dataloader = DataLoader(dataset, batch_size=128, shuffle=True, pin_memory=True)

# modify model
for name, module in m.named_modules():
    if conv2d_predicate(module):
        init_tensor = torch.tensor(0.0)
        # init_tensor.to(device)
        amaxes[name] = init_tensor

        module.register_forward_hook(
            lambda module, inp, out, amax_name=name: get_amax_hook(module, inp, out, amax_name)
        )

m.eval()
with torch.no_grad():
    for X, y_true in tqdm(dataloader):
        X = X.to(device)
        y_prob = m(X)


pp = pprint.PrettyPrinter(indent=4)
pp.pprint(amaxes)

for k, v in amaxes.items():
    amaxes[k] = v.item()

df = pd.DataFrame(data=amaxes, index=[0])
df = (df.T)
df.to_excel('data.xlsx')

