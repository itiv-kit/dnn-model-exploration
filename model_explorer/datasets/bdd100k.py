import os

from torchvision import transforms

from model_explorer.third_party.yolop_det_seg.lib.dataset.bdd import BddDataset
from model_explorer.third_party.yolop_det_seg.lib.config import cfg
from model_explorer.third_party.yolop_det_seg.lib.dataset import AutoDriveDataset


def prepare_bdd100k_dataset(path: str, kind: str, **kwargs):
    transf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cfg.DATASET.DATAROOT = os.path.join(path, kwargs.get('data_root'))
    cfg.DATASET.LABELROOT = os.path.join(path, kwargs.get('label_root'))
    cfg.DATASET.MASKROOT = os.path.join(path, kwargs.get('mask_root'))
    cfg.DATASET.LANEROOT = os.path.join(path, kwargs.get('lane_root'))

    ds = BddDataset(cfg=cfg,
                    is_train=False,
                    inputsize=cfg.MODEL.IMAGE_SIZE,
                    transform=transf)

    return ds


dataset_creator = prepare_bdd100k_dataset
collate_fn = AutoDriveDataset.collate_fn
