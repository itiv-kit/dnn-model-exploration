import torch
from torchvision import transforms, datasets
import random
from torch.utils.data import DataLoader, Subset

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                std=[0.229, 0.224, 0.225])

transforms = transforms.Compose([transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize])

dataset = datasets.ImageFolder('/home/oq4116/temp/ILSVRC/Data/CLS-LOC/val', transforms)
# dataset = datasets.ImageFolder('/data/oq4116/imagenet/val', transforms)
indices = random.sample(range(len(dataset)), 100)
dataset_100 = Subset(dataset, indices=indices)
indices = random.sample(range(len(dataset)), 5000)
dataset_5000 = Subset(dataset, indices=indices)


dev_string = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(dev_string)