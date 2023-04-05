import torch

import model_explorer.third_party.deeplab_v3_pytorch.utils.ext_transforms as et 
from model_explorer.third_party.deeplab_v3_pytorch.datasets.cityscapes import Cityscapes


def prepare_cityscapes_dataset(path: str, kind: str, **kwargs):
    transf = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    split = kwargs.get('split', 'val')
    ds = Cityscapes(root=path, split=split, transform=transf)

    return ds


dataset_creator = prepare_cityscapes_dataset
collate_fn = None
