import torch
import torch.utils.data
from torch import nn
from tqdm import tqdm
import math
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
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model = model.to(device)
model.load_state_dict(torch.load('resnet18-calib.pth', map_location=device))



class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

def get_best_configurations(pkl_file):
    with open(pkl_file, 'rb') as f:
        d = CPU_Unpickler(f).load()
        
    # df_data = np.empty( (0, 111) ) #FIXME: only for resnet50
    df_data = np.empty( (0, 45) ) #FIXME: only for resnet18
    for h in d.history:
        for ind in h.opt:
            l = np.concatenate( (ind.get("G") - 0.65, ind.get("F"), ind.get("X")) )
            l = np.expand_dims(l, axis=0)
            df_data = np.concatenate( (df_data, l), axis=0)
            
    df = pd.DataFrame(df_data)
    df[0] = -df[0] # invert Accuracy
    df_best = df.where(df[0] > 0.65) 
    return df_best


################# Class for Training ###################
class TrainableQuantizationModel(QuantizationModel):
    def __init__(self, model, device, evaluation_samples=None, verbose=False) -> None:
        super().__init__(model, device, evaluation_samples, verbose)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, train_len, val_len, num_epochs=10):
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            loaders = {'train' : train_loader,
                       'val' : val_loader}
            lens = {'train' : math.ceil(train_len / train_loader.batch_size), 
                    'val' : math.ceil(val_len / val_loader.batch_size)}

            for phase in ['val', 'train']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0
                running_corrects = 0

                for image, target in tqdm(loaders[phase], 
                        desc='Epoch: {}'.format(epoch), 
                        total=lens[phase]):
                    image, target = image.to(device), target.to(device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        output = self.model(image)
                        _, preds = torch.max(output, 1)
                        loss = self.criterion(output, target)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    running_loss += loss.item() * image.size(0)
                    running_corrects += torch.sum(preds == target.data)

                if phase == 'train':
                    self.lr_scheduler.step()
                
                epoch_loss = running_loss / lens[phase]
                epoch_acc = running_corrects.float() / lens[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                print()

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)


best = get_best_configurations('exploration_resnet18.pkl')

val_dataloader = get_dataloader(batch_size=64, dataset='val')
train_dataloader = get_dataloader(batch_size=64, dataset='train')

layernames = []
for name, module in model.named_modules():
    if isinstance(module, quant_nn.TensorQuantizer):
        layernames.append(name)


print("Testing {} Iteration(s)".format(len(best)))

for index, row in best.iterrows():
    if any(row.isna()):
        continue 
    m = TrainableQuantizationModel(model, device, verbose=True)
    bitw = {}
    for i, name in enumerate(layernames):
        bitw[name] = int(row[2+i])
    m.bit_widths = bitw
    print("#"*100)
    print("{} / {}: Running: " + ", ".join(index, len(best), map(str, row)))
    print("#"*100)
    
    # train one epoch
    m.train(
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        train_len=1281167,
        val_len=50000,
        num_epochs=3
    )

    acc2 = m.evaluate(val_dataloader)
    # print("Acc after retraining: {}".format(acc2))
        
    torch.save(model.state_dict(), 'model_{}.m'.format(index))
        