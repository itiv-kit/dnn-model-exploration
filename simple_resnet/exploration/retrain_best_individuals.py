import torch
import torch.utils.data
from torch import nn
from tqdm import tqdm
import pickle, io

import numpy as np
import pandas as pd

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor

from torchvision import models
from torch.utils.data import DataLoader

from pytorch_quantization import quant_modules
quant_modules.initialize()

from commons import get_dataloader, device

from explore import LayerwiseQuantizationProblem, QuantizationModel

#load model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model = model.to(device)
model.load_state_dict(torch.load('resnet50-calib.pth', map_location=device))



class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

def get_best_configurations(pkl_file):
    with open(pkl_file, 'rb') as f:
        d = CPU_Unpickler(f).load()
        
    df_data = np.empty( (0, 111) ) #FIXME: only for resnet
    for h in d.history:
        for ind in h.opt:
            l = np.concatenate( (ind.get("G") - 0.75, ind.get("F"), ind.get("X")) )
            l = np.expand_dims(l, axis=0)
            df_data = np.concatenate( (df_data, l), axis=0)
            
    df = pd.DataFrame(df_data)
    df[0] = -df[0] # invert Accuracy
    df_best = df.where(df[0] > 0.75) 
    return df_best


################# Class for Training ###################
class TrainableQuantizationModel(QuantizationModel):
    def __init__(self, model, device, evaluation_samples=None, verbose=False) -> None:
        super().__init__(model, device, evaluation_samples, verbose)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)
    
    def train(self, dataloader: DataLoader, epochs=10):
        for i in range(epochs):
            self._train_one_epoch(desc=str(i), dataloader=dataloader)
    
    def _train_one_epoch(self, dataloader: DataLoader, desc=None):
        self.model.train()
        
        for image, target in tqdm(dataloader, desc='Epoch: {}'.format(desc)):
            image, target = image.to(device), target.to(device)
            output = self.model(image)
            loss = self.criterion(output, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)


best = get_best_configurations('exploration_15_15_12bit_weighted.pkl')

val_dataloader = get_dataloader(method='all', webdataset=False, batch_size=16, dataset='val')
train_dataloader = get_dataloader(method='all', webdataset=True, batch_size=16, dataset='train')

layernames = []
for name, module in model.named_modules():
    if isinstance(module, quant_nn.TensorQuantizer):
        layernames.append(name)


for index, row in best.iterrows():
    if any(row.isna()):
        continue 
    m = TrainableQuantizationModel(model, device, verbose=True)
    bitw = {}
    for i, name in enumerate(layernames):
        bitw[name] = int(row[2+i])
    m.bit_widths = bitw
    print("Running: " + ", ".join(map(str, row)))
    m.evaluate(val_dataloader)
    
    # train one epoch
    m.train(train_dataloader, epochs=3)

    m.evaluate(val_dataloader)
        
    torch.save(model.state_dict(), 'model_{}.m'.format(index))
        