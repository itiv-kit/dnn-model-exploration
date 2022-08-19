import sys
import pickle
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

import numpy as np 
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.termination import get_termination

from torchinfo import summary

from pytorch_quantization import quant_modules
quant_modules.initialize()

from commons import device, dataset, dataset_100, dataset_5000

#load model:
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.to(device)
model.load_state_dict(torch.load('resnet50-calib.pth', map_location=device))


class QuantizationModel(nn.Module):
    def __init__(self, model, device) -> None:
        super().__init__()

        self.model = model
        self.bit_widths = {}
        self.device = device

    def set_bit_widths(self, bit_widths):
        self.bit_widths = bit_widths

    def prepare_model(self):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                assert name in self.bit_widths, "Layer {} not found in bit_widths dict".format(name)
                module.num_bits = self.bit_widths[name]

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


class LayerwiseQuantizationProblem(ElementwiseProblem):
    def __init__(self, q_model, dataloader, n_var, layernames, **kwargs):
        super().__init__(
            n_var=n_var,
            n_obj=2,
            n_constr=1,
            xl=2,
            xu=16,
            vtype=int,
            **kwargs)
        
        self.q_model = q_model
        self.dataloader = dataloader
        self.layernames = layernames

    def _evaluate(self, x, out, *args, **kwargs):
        bit_widths = {}
        for i, name in enumerate(self.layernames):
            bit_widths[name] = int(x[i])
        self.q_model.set_bit_widths(bit_widths)

        self.q_model.prepare_model()
        f1_acc = self.q_model.evaluate(self.dataloader)
        f2_bits = np.sum(x)
        print("acc of pass {}%".format(f1_acc * 100))
        g1_acc_constraint = 0.70 - f1_acc
        out["F"] = [-f1_acc, f2_bits]
        out["G"] = [g1_acc_constraint]


# get layernames
layernames = []
for name, module in model.named_modules():
    if isinstance(module, quant_nn.TensorQuantizer):
        layernames.append(name)

# explore
qmodel = QuantizationModel(model, device)

dataloader = DataLoader(dataset_5000, batch_size=64, shuffle=True, pin_memory=True)
problem = LayerwiseQuantizationProblem(
    q_model=qmodel,
    dataloader=dataloader,
    n_var=len(layernames),
    layernames=layernames
)


sampling = IntegerRandomSampling()
crossover = SBX(prob_var=1.0, repair=RoundingRepair(), vtype=float)
mutation = PolynomialMutation(prob=1.0, repair=RoundingRepair())

algorithm = NSGA2(
    pop_size=20,
    n_offsprings=20,
    sampling=sampling,
    crossover=crossover,
    mutation=mutation,
    eliminate_duplicates=True)

termination = get_termination("n_gen", 30)

res = minimize(
    problem,
    algorithm,
    termination,
    verbose=True
)

with open('exploration.pkl', 'wb') as f:
    pickle.dump(res, f)



