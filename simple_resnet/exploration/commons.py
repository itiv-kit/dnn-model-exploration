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


def get_dataloader(batch_size=64, dataset='val'):
    assert dataset in ['val', 'test', 'train'] 

    webdatasets = {'val'   : '/tools/datasets/imagenet/val/imagenet-val-{0000..0006}.tar', 
                   'train' : '/tools/datasets/imagenet/train/imagenet-train-{0000..0136}.tar'}
    dataset_lens = {'val' : 50000,
                    'train' : 1281167}

    transforms = get_transforms()
    
    dataset_load = (
        wds.WebDataset(webdatasets[dataset])
        .decode("pil")
        .to_tuple("input.jpeg", "target.cls")
        .shuffle(10000)
        .map_tuple(transforms, identity)
    )
    
    return DataLoader(dataset_load, batch_size=batch_size, pin_memory=True)


def get_val_dataset_raw(method='true_random', batch_size=64, samples=None):
    assert method in ['true_random', 'fixed_random_selection', 'all']

    transforms = get_transforms()
    
    path = '/tools/datasets/imagenet/val_images'
    path = '/data/oq4116/imagenet/val'
    dataset_load = datasets.ImageFolder(path, transforms)

    if method == 'true_random':
        # with a random sampler dataset gets shuffeld each time
        data_sampler = RandomSampler(dataset_load)
        return DataLoader(dataset_load, batch_size=batch_size, 
                          pin_memory=True, sampler=data_sampler)
    elif method == 'fixed_random_selection':
        indices = random.sample(range(len(dataset_load)), samples)
        subset_ds = Subset(dataset_load, indices=indices)
        return DataLoader(subset_ds, batch_size=batch_size,
                          pin_memory=True)
    elif method == 'all':
        return DataLoader(dataset_load, batch_size=batch_size, pin_memory=True)



dev_string = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(dev_string)
