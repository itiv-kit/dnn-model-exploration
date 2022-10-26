
import math
import pickle
import torch
import torch.utils.data
from torch import nn
import random
from tqdm import tqdm

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor

from torchvision import models
from torch.utils.data import DataLoader

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

from commons import device, get_dataloader, get_val_dataset_raw

#load model:
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.to(device)
model.load_state_dict(torch.load('resnet18-calib.pth', map_location=device))


class QuantizationModel(nn.Module):
    def __init__(self, model, device, dataset_len, evaluation_samples=None, verbose=False) -> None:
        super().__init__()

        self.model = model
        self._bit_widths = {}
        self.device = device
        self.dataset_len = dataset_len
        self.evaluation_samples = evaluation_samples
        self.verbose = verbose

    @property
    def bit_widths(self):
        return self._bit_widths

    @bit_widths.setter
    def bit_widths(self, new_bit_widths):
        assert isinstance(new_bit_widths, dict), "bit_width have to be a dict"

        # Update Model ...
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                assert name in new_bit_widths, "Layer {} not found in bit_widths dict".format(name)
                module.num_bits = new_bit_widths[name]

        self._bit_widths = new_bit_widths

    def evaluate(self, dataloader: DataLoader):
        count = self.evaluation_samples if self.evaluation_samples is not None else self.dataset_len

        correct_pred = torch.tensor(0).to(self.device)
        self.model.eval()
        self.model = self.model.to(self.device)
        with torch.no_grad():
            for i, (X, y_true) in tqdm(enumerate(dataloader), 
                    disable=not self.verbose, 
                    total=math.ceil(count/dataloader.batch_size)):
                X = X.to(self.device)
                y_true = y_true.to(self.device)

                y_prob = self.model(X)
                _, predicted_labels = torch.max(y_prob, 1)

                correct_pred += (predicted_labels == y_true).sum()

                if self.evaluation_samples is not None:
                    if i * dataloader.batch_size > self.evaluation_samples:
                        break

        # total_samples = math.ceil(self.evaluation_samples / dataloader.batch_size) * dataloader.batch_size
        total_samples = i * dataloader.batch_size
        accuracy = correct_pred.float() / total_samples
        return accuracy


class LayerwiseQuantizationProblem(ElementwiseProblem):
    def __init__(self, 
            q_model:QuantizationModel, 
            dataloader:DataLoader, 
            n_var:int, 
            layernames,
            **kwargs):
        super().__init__(
            n_var=n_var,
            n_obj=1,
            n_constr=1,
            xl=2,
            xu=12,
            vtype=int,
            **kwargs)

        self.q_model = q_model
        self.dataloader = dataloader
        self.layernames = layernames
        self.cpu_device = torch.device("cpu")

    def _evaluate(self, x, out, *args, **kwargs):
        bit_widths = {}
        factors = np.zeros(x.shape)
        for i, name in enumerate(self.layernames):
            bit_widths[name] = int(x[i])
            if 'weight' in name:
                factors[i] = 0.5
            elif 'input' in name:
                factors[i] = 5.0
            else:
                factors[i] = 1.0
        self.q_model.bit_widths = bit_widths

        f1_acc = self.q_model.evaluate(self.dataloader).to(self.cpu_device)
        f2_bits = np.sum(x * factors)
        print("acc of pass {:.4f}% with {} weighted bits".format(f1_acc * 100, f2_bits))
        g1_acc_constraint = 0.65 - f1_acc
        out["F"] = [f2_bits]
        out["G"] = [g1_acc_constraint]


if __name__ == "__main__":
    # get layernames
    layernames = []
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            layernames.append(name)

    total_bits = 0
    max_bits = 12.0
    for i, name in enumerate(layernames):
        if 'weight' in name:
            total_bits += 0.5 * max_bits
        elif 'input' in name:
            total_bits += 5.0 * max_bits
        else:
            total_bits += 1.0 * max_bits
    print("Max weighted bits at {} bit per layer: {:.1f}".format(int(max_bits), total_bits))

    # explore
    qmodel = QuantizationModel(model, device, dataset_len=50000, evaluation_samples=5000, verbose=True)

    # dataloader = get_dataloader(batch_size=64, dataset='val')
    dataloader = get_val_dataset_raw(method='true_random', batch_size=512, samples=5000)

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
        pop_size=15,
        n_offsprings=15,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True)

    termination = get_termination("n_gen", 15)

    res = minimize(
        problem,
        algorithm,
        termination,
        verbose=True,
        save_history=True
    )

    with open('exploration_resnet18.pkl', 'wb') as f:
        pickle.dump(res, f)



