import torch
from model_explorer.workloads.unet import UNet

fn = 'resnet18_unet.pth'

m = UNet(fs=32, expansion=1, n_out=13)
state_dict = torch.load(fn)
m.load_state_dict(state_dict)

from torchinfo import summary
summary(m, (1, 3, 608, 800), depth=100)


input_file = '/home/oq4116/temp/unet-image-segmentation/data/Images/Video_021/v021_0233.png'
from PIL import Image
from torchvision import transforms

input_image = Image.open(input_file)
input_image = input_image.convert("RGB")
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

output = m(input_batch)

palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = torch.as_tensor([i for i in range(13)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")

# plot the semantic segmentation predictions of 21 classes in each color
r = Image.fromarray(output.byte().cpu().numpy()).resize(input_image.size)
r.putpalette(colors)

import matplotlib.pyplot as plt
plt.imshow(r)



