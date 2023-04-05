# DAVID Dataset from TUM
import glob
import os
import torch

from torchvision import transforms
import torch.utils.data as data
from PIL import Image
import matplotlib.image as mpimg


class SegmentationDataset(data.Dataset):
    """Helper class to load a segmentation dataset, adopted from this
    discussion:
    https://discuss.pytorch.org/t/dataloader-for-semantic-segmentation/48290
    """

    def __init__(self, folder_path, transforms, data_folder="Images", labels_folder="Labels"):
        super(SegmentationDataset, self).__init__()
        self.img_files = glob.glob(os.path.join(folder_path, data_folder, '*.png'))
        self.mask_files = []
        self.transforms = transforms
        for img_path in self.img_files:
            self.mask_files.append(os.path.join(folder_path, labels_folder, os.path.basename(img_path)))

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]

        data_img = Image.open(img_path).convert("RGB")
        data = self.transforms(data_img)

        label = mpimg.imread(mask_path) * 255
        return data, torch.from_numpy(label).float()

    def __len__(self):
        return len(self.img_files)


def prepare_david_dataset(path: str, kind: str, **kwargs):
    """Load a DAVID dataset (which is provided by TU Munich)

    Args:
        path (str): Location of the dataset
        kind (str): not used

    Returns:
        dataset: loaded dataset
    """
    transf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((608, 800))
    ])

    ds = SegmentationDataset(path, transf, 'images_raw', 'labels_int')

    return ds


dataset_creator = prepare_david_dataset
collate_fn = None
