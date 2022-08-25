import torch
import torch.utils.data
from torch import nn
from tqdm import tqdm

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor

from torchvision import models

from pytorch_quantization import quant_modules
quant_modules.initialize()

from commons import dataset, device


# quant_desc_input = QuantDescriptor(calib_method='histogram')
# quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
# quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, pin_memory=True, shuffle=True)

#load model:
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model = model.to(device)
model.load_state_dict(torch.load('resnet50-calib.pth', map_location=device))


# test top 1 
def evaluate(model):
    correct_pred = torch.tensor(0).to(device)
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        for i, (X, y_true) in tqdm(enumerate(dataloader)):
            X = X.to(device)
            y_true = y_true.to(device)

            y_prob = model(X)
            _, predicted_labels = torch.max(y_prob, 1)

            correct_pred += (predicted_labels == y_true).sum()

    accuracy = correct_pred.float() / len(dataset)
    print("Accuracy is {:.4f}%".format(accuracy*100))


evaluate(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# train one epoch
model.train()
for image, target in tqdm(dataloader):
    image, target = image.to(device), target.to(device)
    output = model(image)
    loss = criterion(output, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


evaluate(model)
    
    
    