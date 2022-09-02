from numpy import identity
import torch
from torchvision import transforms, datasets
import random
from torch.utils.data import DataLoader, Subset, RandomSampler
import os
import socket
import webdataset as wds


def get_transforms():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                    std=[0.229, 0.224, 0.225])

    tfs = transforms.Compose([transforms.Resize(256),
                              transforms.CenterCrop(224),
                              transforms.ToTensor(),
                              normalize])

    return tfs

def identity(d):
    return d

def get_dataloader(method='true_random', webdataset=True, batch_size=64, dataset='val', samples=None):
    assert method in ['true_random', 'fixed_random_selection', 'all']
    assert dataset in ['val', 'test', 'train'] 

    webdatasets = {'val'   : '/tools/datasets/imagenet/val/imagenet-val-{0000..0032}.tar', 
                   'train' : '/tools/datasets/imagenet/train/imagenet-train-{0000..0136}.tar'}

    transforms = get_transforms()
    
    if webdataset:
        dataset = (
            wds.WebDataset(webdatasets[dataset])
            .decode("pil")
            .to_tuple("input.jpeg", "target.cls")
            .map_tuple(transforms, identity)
        )
    else:
        if socket.gethostname() == 'itiv-work5.itiv.kit.edu' or socket.gethostname().startswith('itiv-pool'):
            path = '/home/oq4116/temp/ILSVRC/Data/CLS-LOC'
        elif socket.gethostname() == 'titanv':
            path = '/data/oq4116/imagenet/val'
        else:
            raise ValueError("Invalid host ...")
        dataset = datasets.ImageFolder(os.path.join(path, dataset), transforms)

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
