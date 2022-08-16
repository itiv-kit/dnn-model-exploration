import pprint
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from torchvision import models
from torchvision import transforms
from torchvision import datasets

from torchinfo import summary

from pytorch_quantization.tensor_quant import QuantDescriptor
import pytorch_quantization.nn as quant_nn
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer

from tqdm import tqdm
import pandas as pd


BITS = 8


def conv2d_predicate(module):
    return isinstance(module, torch.nn.Conv2d)

class CalibrationModel(nn.Module):
    def __init__(self, model, device):
        super().__init__()

        self.model = model
        self.metrics = {}
        self.device = device
        self.hook_handles = []

        self._prepare_model()

    def _prepare_model(self):
        for name, module in self.model.named_modules():
            if conv2d_predicate(module):
                init_tensor_min = torch.tensor(float('inf'))
                init_tensor_max = torch.tensor(float('-inf'))
                
                self.metrics[name] = {}
                self.metrics[name]['max'] = init_tensor_max
                self.metrics[name]['min'] = init_tensor_min

                handle = module.register_forward_hook(
                    lambda module, inp, out, module_name=name: self.get_amax_hook(module, inp, out, module_name)
                )
                self.hook_handles.append(handle)

    def get_amax_hook(self, module, inp, outp, module_name):
        cpu_dev = torch.device("cpu")

        amax = torch.amax(inp[0]).to(cpu_dev)
        amin = torch.amin(inp[0]).to(cpu_dev)
        prev_amax = self.metrics[module_name]['max']
        prev_amin = self.metrics[module_name]['min']

        compare_max = torch.stack((amax, prev_amax))
        compare_min = torch.stack((amin, prev_amin))
        self.metrics[module_name]['max'] = torch.amax(compare_max)
        self.metrics[module_name]['min'] = torch.amin(compare_min)

    def generate_metrics(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            for X, _ in tqdm(dataloader):
                X = X.to(self.device)
                _ = self.model(X)

        for k, v in self.metrics.items():
            for k2, v2 in v.items():
                self.metrics[k][k2] = v2.item()

    def get_metrics(self):
        return self.metrics
            
    def store_metrics(self, xls_file='data.xlsx'):
        df = pd.DataFrame(data=self.metrics, index=[0])
        df = (df.T)
        df.to_excel(xls_file)

    def print_metrics(self):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.metrics)

    def restore_model(self):
        for hook in self.hook_handles:
            hook.remove()


class QuantizationModel(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()

        self.model = model
        self.bit_width = None
        self.hook_handles = []
        self.metrics = {}

    def set_bit_width(self, bit_width):
        self.bit_width = bit_width

    def set_metrics(self, metrics):
        self.metrics = metrics

    def fake_quantizer_hook(self, module, inp, quant):
        return quant(inp[0])

    def prepare_model(self):
        for name, module in m.named_modules():
            if conv2d_predicate(module):
                quant_desc = QuantDescriptor(
                    num_bits=self.bit_width,
                    fake_quant=True,
                    axis=None,
                    unsigned=False,
                    amax=self.metrics[name]['max']
                )
                fake_quantizer = TensorQuantizer(quant_desc)
                fake_quantizer = fake_quantizer.to(device)

                hook = module.register_forward_pre_hook(
                    lambda module, inp, quant=fake_quantizer: \
                        self.fake_quantizer_hook(module, inp, quant)
                )
                self.hook_handles.append(hook)

    def evaluate(self, dataloader):
        correct_pred = 0
        self.model.eval()
        self.model = self.model.to(device)
        with torch.no_grad():
            for X, y_true in tqdm(dataloader):
                X = X.to(device)
                y_true = y_true.to(device)

                y_prob = self.model(X)
                _, predicted_labels = torch.max(y_prob, 1)

                correct_pred += (predicted_labels == y_true).sum()

        accuracy = correct_pred.float() / len(dataloader.dataset)
        return accuracy

    def restore_model(self):
        for hook in self.hook_handles:
            hook.remove()



dev_string = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(dev_string)

m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
m.to(device)

# prepare dataset
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

transforms = transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               normalize])

# dataset = datasets.ImageFolder('/data/oq4116/imagenet/val', transforms)
dataset = datasets.ImageFolder('/home/oq4116/temp/ILSVRC/Data/CLS-LOC/val', transforms)
indices = random.sample(range(len(dataset)), 1000)
dataset_1000 = Subset(dataset, indices=indices)
indices = random.sample(range(len(dataset)), 100)
dataset_100 = Subset(dataset, indices=indices)


print("Starting calibration...")

dataloader = DataLoader(dataset_1000, batch_size=64, shuffle=True, pin_memory=True)
cm = CalibrationModel(m, device)
cm.generate_metrics(dataloader)
metrics = cm.get_metrics()
cm.restore_model()

print("Calibration done ...")

print("Running Inference at 7 bits")

dataloader = DataLoader(dataset_100, batch_size=64, shuffle=True, pin_memory=True)
qm = QuantizationModel(m)
qm.set_bit_width(bit_width=7)
qm.set_metrics(metrics)
qm.prepare_model()
acc = qm.evaluate(dataloader)
qm.restore_model()
print("Accuracy at 7 bits is {:.4f}%".format(acc * 100))
qm.set_bit_width(bit_width=4)
qm.prepare_model()
acc = qm.evaluate(dataloader)
print("Accuracy at 4 bits is {:.4f}%".format(acc * 100))


