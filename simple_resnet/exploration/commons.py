import torch
from torchvision import transforms, datasets
import random
from torch.utils.data import DataLoader, Subset, RandomSampler


import socket

def get_transforms():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                    std=[0.229, 0.224, 0.225])

    tfs = transforms.Compose([transforms.Resize(256),
                              transforms.CenterCrop(224),
                              transforms.ToTensor(),
                              normalize])

    return tfs

def get_dataloader(method='true_random', batch_size=64, samples=None):
    assert method in ['true_random', 'fixed_random_selection', 'all']

    transforms = get_transforms()

    if socket.gethostname() == 'itiv-work5.itiv.kit.edu' or socket.gethostname().startswith('itiv-pool'):
        dataset = datasets.ImageFolder('/home/oq4116/temp/ILSVRC/Data/CLS-LOC/val', transforms)
    elif socket.gethostname() == 'titanv':
        dataset = datasets.ImageFolder('/data/oq4116/imagenet/val', transforms)
    else:
        raise ValueError("Invalid host ...")

    if method == 'true_random':
        # with a random sampler dataset gets shuffeld each time
        data_sampler = RandomSampler(dataset)
        return DataLoader(dataset, batch_size=batch_size, 
                          pin_memory=True, sampler=data_sampler)
    elif method == 'fixed_random_selection':
        indices = random.sample(range(len(dataset)), samples)
        subset_ds = Subset(dataset, indices=indices)
        return DataLoader(subset_ds, batch_size=batch_size,
                          pin_memory=True)
    elif method == 'all':
        return DataLoader(dataset, batch_size=batch_size, pin_memory=True)


dev_string = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(dev_string)
