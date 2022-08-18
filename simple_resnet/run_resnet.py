import pprint
import random
import pickle 
import os

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

import numpy as np 
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.termination import get_termination


def conv2d_predicate(module):
    return isinstance(module, torch.nn.Conv2d)

class CalibrationModel(nn.Module):
    def __init__(self, model, device):
        super().__init__()

        self.model = model
        self.metrics = {}
        self.device = device
        self.hook_handles = []
        self.matching_layer_count = 0

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
                self.matching_layer_count += 1

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
        if not self.metrics:
            print("Empty Metrics, run generate_metrics first...")
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
    def __init__(self, model, device) -> None:
        super().__init__()

        self.model = model
        self.device = device
        self.bit_widths = {}
        self.hook_handles = []
        self.metrics = {}

    def set_bit_widths(self, bit_widths):
        self.bit_widths = bit_widths

    def set_metrics(self, metrics):
        self.metrics = metrics

    def fake_quantizer_hook(self, module, inp, quant):
        return quant(inp[0])

    def prepare_model(self):
        for name, module in m.named_modules():
            if conv2d_predicate(module):
                assert name in self.bit_widths, "Layer {} not found in bit_widths dict".format(name)
                quant_desc = QuantDescriptor(
                    num_bits=self.bit_widths[name],
                    fake_quant=True,
                    axis=None,
                    unsigned=False,
                    amax=self.metrics[name]['max']
                )
                fake_quantizer = TensorQuantizer(quant_desc)
                fake_quantizer = fake_quantizer.to(self.device)

                hook = module.register_forward_pre_hook(
                    lambda module, inp, quant=fake_quantizer: \
                        self.fake_quantizer_hook(module, inp, quant)
                )
                self.hook_handles.append(hook)

    def evaluate(self, dataloader):
        correct_pred = 0
        self.model.eval()
        self.model = self.model.to(self.device)
        with torch.no_grad():
            for X, y_true in tqdm(dataloader):
                X = X.to(self.device)
                y_true = y_true.to(self.device)

                y_prob = self.model(X)
                _, predicted_labels = torch.max(y_prob, 1)

                correct_pred += (predicted_labels == y_true).sum()

        accuracy = correct_pred.float() / len(dataloader.dataset)
        return accuracy

    def restore_model(self):
        for hook in self.hook_handles:
            hook.remove()

class LayerwiseQuantizationProblem(ElementwiseProblem):
    def __init__(self, q_model, dataloader, n_var, layernames, **kwargs):
        super().__init__(
            n_var=n_var,
            n_obj=2,
            n_constr=1,
            xl=2,
            xu=14,
            vtype=int,
            **kwargs)

        self.q_model = q_model
        self.dataloader = dataloader
        self.layernames = layernames

        self.cpu_device = torch.device("cpu")

    def _evaluate(self, x, out, *args, **kwargs):
        bit_widths = {}
        for i, name in enumerate(self.layernames):
            bit_widths[name] = int(x[i])
        self.q_model.set_bit_widths(bit_widths)

        self.q_model.prepare_model()
        f1_acc = self.q_model.evaluate(self.dataloader).to(self.cpu_device)
        f2_bits = np.sum(x)

        print(f1_acc.device)
        print("acc of pass {}%".format(f1_acc * 100))
        g1_acc_constraint = 0.60 - f1_acc
        out["F"] = [-f1_acc, f2_bits]
        out["G"] = [g1_acc_constraint]
        self.q_model.restore_model()


dev_string = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(dev_string)

m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
m.to(device)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

transforms = transforms.Compose([transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize])

dataset = datasets.ImageFolder('/data/oq4116/imagenet/val', transforms)
# dataset = datasets.ImageFolder('/home/oq4116/temp/ILSVRC/Data/CLS-LOC/val', transforms)
indices = random.sample(range(len(dataset)), 10000)
dataset_10000 = Subset(dataset, indices=indices)
indices = random.sample(range(len(dataset)), 5000)
dataset_5000 = Subset(dataset, indices=indices)
indices = random.sample(range(len(dataset)), 1000)
dataset_1000 = Subset(dataset, indices=indices)
indices = random.sample(range(len(dataset)), 100)
dataset_100 = Subset(dataset, indices=indices)


# prepare dataset and check if calibration is already present
metrics = {}
if os.path.exists('calibration_resnet50.pkl'):
    with open('calibration_resnet50.pkl', 'rb') as f:
        metrics = pickle.load(f)
    print("Calibration read ...")

else:
    print("Starting calibration...")

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True)
    cm = CalibrationModel(m, device)
    cm.generate_metrics(dataloader)
    metrics = cm.get_metrics()
    cm.restore_model()

    with open('calibration_resnet50.pkl', 'wb') as f:
        pickle.dump(metrics, f)

    print("Calibration done ...")

layernames = [name for name, _ in metrics.items()]
layercount = len(metrics)

dataloader = DataLoader(dataset_5000, batch_size=64, shuffle=True, pin_memory=True)
qm = QuantizationModel(m, device=device)
qm.set_metrics(metrics)


problem = LayerwiseQuantizationProblem(
    qm, dataloader, layercount, layernames
)

sampling = IntegerRandomSampling()
crossover = SBX(prob_var=1.0, repair=RoundingRepair(), vtype=float)
mutation = PolynomialMutation(prob=1.0, repair=RoundingRepair())

algorithm = NSGA2(
    pop_size=30,
    n_offsprings=30,
    sampling=sampling,
    crossover=crossover,
    mutation=mutation,
    eliminate_duplicates=True)

termination = get_termination("n_gen", 20)

res = minimize(
    problem,
    algorithm,
    termination,
    verbose=True
)

with open('exploration.pkl', 'wb') as f:
    pickle.dump(res, f)



