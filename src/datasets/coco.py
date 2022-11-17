import torchvision
import torch

def coco_collate_fn(batch):
    """Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
    This describes how to combine these tensors of different sizes. We use lists.

    Args:
        batch (tuple): an iterable of N sets from __getitem__()

    Returns:
        list: a tensor of images, list of targets
    """

    images = list()
    targets = list()

    for b in batch:
        images.append(b[0])
        targets.append(b[1])
    
    if isinstance(images[0], torch.Tensor):
        images = torch.stack(images, dim=0)
    
    return images, targets


def prepare_coco_dataset(path, transforms, groundtruth, **kwargs):
    """Load a coco detection dataset.

    Args:
        path (str): Root directory for the images.
        transforms (callable): A function/transform that takes input sample and its target as entry
        and returns a transformed version.
        groundtruth (str): Path to json annotation file.

    Returns:
        torchvision.datasets.CocoDetection: The loaded dataset.
    """
    dataset = torchvision.datasets.CocoDetection(path, groundtruth,
        transforms=transforms)

    return dataset


collate_fn = coco_collate_fn
get_validation_dataset = prepare_coco_dataset
get_train_dataset = None
