"""Modified version of Cityscapes loader class, but for different weather conditions
"""
import json
import os
from collections import namedtuple

import torch.utils.data as data
from PIL import Image
import numpy as np

import model_explorer.third_party.deeplab_v3_pytorch.utils.ext_transforms as et


class CityscapesWithWeather(data.Dataset):
    """Cityscapes <http://www.cityscapes-dataset.com/> Dataset.

    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    CityscapesWeatherConditions = namedtuple('CityscapesWeatherConditions', [
                                             'name', 'path_extra', 'file_extra', 'parameter', 'parameter_steps'])
    classes = [
        CityscapesClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence',                13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
    ]
    weather_conditions = [
        CityscapesWeatherConditions('rain', 'rain', 'rain', ['alpha', 'beta', 'dropsize', 'pattern'], [[0.01, 0.02, 0.03], [
                                    0.005, 0.01, 0.015], [0.002, 0.005, 0.01], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]),
        CityscapesWeatherConditions('fog', 'foggyDBF', 'foggy', ['beta'], [[0.005, 0.01, 0.02]])
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])

    def __init__(self, root, weather_condition, weather_condition_kwparams, split='train', target_type='semantic', transform=None):
        self.root = os.path.expanduser(root)
        self.mode = 'gtFine'
        self.target_type = target_type

        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.transform = transform

        self.split = split
        self.images = []
        self.targets = []

        if weather_condition is not None:
            self._select_weather_condition(weather_condition, weather_condition_kwparams)
            self.images_dir = os.path.join(self.root, f'leftImg8bit_{self.wc_path_extra}', split)
        else:
            self.images_dir = os.path.join(self.root, 'leftImg8bit', split)

        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)

            for file_name in os.listdir(img_dir):
                # check for weather condition data:
                if weather_condition is not None:
                    file_ending = f"_{self.wc_file_extra}_"
                    file_ending += "_".join([str(item) for pair in zip(self.wc_parameter_k, self.wc_parameter_v) for item in pair])
                    file_ending += ".png"
                else:
                    # no weather condition given
                    file_ending = ".png"

                if file_name.endswith(file_ending):
                    self.images.append(os.path.join(img_dir, file_name))

                    target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                                    self._get_target_suffix(self.mode, self.target_type))
                    self.targets.append(os.path.join(target_dir, target_name))


    def _select_weather_condition(self, name: str, kwparams):
        condition = [wc for wc in self.weather_conditions if wc.name == name][0]
        self.wc_path_extra = condition.path_extra
        self.wc_file_extra = condition.file_extra
        self.wc_parameter_k = []
        self.wc_parameter_v = []

        for i, key in enumerate(condition.parameter):
            # we need it in any case
            self.wc_parameter_k.append(key)
            if key in kwparams.keys():
                if kwparams[key] in condition.parameter_steps[i]:
                    self.wc_parameter_v.append(kwparams[key])
                else:
                    self.wc_parameter_v.append(condition.parameter_steps[i][0])
            else:
                self.wc_parameter_v.append(condition.parameter_steps[i][0])

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        # target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])
        if self.transform:
            image, target = self.transform(image, target)
        target = self.encode_target(target)
        return image, target

    def __len__(self):
        return len(self.images)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        elif target_type == 'polygon':
            return '{}_polygons.json'.format(mode)
        elif target_type == 'depth':
            return '{}_disparity.png'.format(mode)


def prepare_cityscapes_weather_dataset(path: str, kind: str, **kwargs):
    """Load cityscapes dataset, defaults to 'gtFine' subset

    Args:
        path (str): Location of the dataset
        kind (): not used

    Returns:
        dataset: Loaded dataset
    """
    transf = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    split = kwargs.get('split', 'val')
    condition = kwargs.get('weather_condition', None)
    ds = CityscapesWithWeather(root=path, weather_condition=condition,
                               weather_condition_kwparams=kwargs, split=split, transform=transf)

    return ds


dataset_creator = prepare_cityscapes_weather_dataset
collate_fn = None
