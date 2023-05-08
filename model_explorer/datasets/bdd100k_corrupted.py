import os

from torchvision import transforms

from model_explorer.third_party.yolop_det_seg.lib.dataset.bdd import BddDataset
from model_explorer.third_party.yolop_det_seg.lib.config import cfg
from model_explorer.third_party.yolop_det_seg.lib.dataset import AutoDriveDataset


def prepare_bdd100k_dataset_corrupted(path: str, kind: str, **kwargs):
    transf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    severity = kwargs.get('severity', 1)
    condition = kwargs.get('condition', 'fog')
    assert severity in range(1, 6)
    assert condition in ['fog', 'frost', 'snow', 'brightness']

    cfg.DATASET.DATAROOT = os.path.join(path, kwargs.get('data_root'), condition, str(severity))
    cfg.DATASET.LABELROOT = os.path.join(path, kwargs.get('label_root'))
    cfg.DATASET.MASKROOT = os.path.join(path, kwargs.get('mask_root'))
    cfg.DATASET.LANEROOT = os.path.join(path, kwargs.get('lane_root'))
    cfg.DATASET.TEST_SET = ''
    cfg.DATASET.TRAIN_SET = ''

    ds = BddDataset(cfg=cfg,
                    is_train=False,
                    inputsize=cfg.MODEL.IMAGE_SIZE,
                    transform=transf)

    return ds


dataset_creator = prepare_bdd100k_dataset_corrupted
collate_fn = AutoDriveDataset.collate_fn
