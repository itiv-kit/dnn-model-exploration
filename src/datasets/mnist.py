from torch.utils.data import DataLoader
from torchvision import datasets

from transforms.mnist_transforms import transforms as mnist_trafos


BATCH_SIZE = 32

def get_train_loader(transform):
    """Get a mnist training data loader.

    Args:
        transforms (callable): A function/transform that takes input sample and its target as entry
        and returns a transformed version.

    Returns:
        DataLoader: The data loader for the mnist training data.
    """
    # download and create datasets
    train_dataset = datasets.MNIST(root='mnist_data',
                                   train=True,
                                   transform=transform,
                                   download=True)

    # define the data loaders
    return DataLoader(dataset=train_dataset,
                      batch_size=BATCH_SIZE,
                      shuffle=True)


def get_validation_loader(download=False):
    """Get a mnist validation data loader.

    Args:
        download (bool): Wether to download the the mnist dataset.

    Returns:
        DataLoader: The data loader for the mnist validation data.
    """
    valid_dataset = prepare_mnist_dataset("mnist_data", mnist_trafos, download)

    return DataLoader(dataset=valid_dataset,
                      batch_size=BATCH_SIZE,
                      shuffle=False)


def prepare_mnist_dataset(path, transforms, download=False, **kwargs):
    """Load a mnist dataset.

    Args:
        path (str): Root directory for the mnist images.
        transforms (callable): A function/transform that takes input sample and its target as entry
        and returns a transformed version.
        download (bool): Wether to download the mnist dataset.

    Returns:
        datasets: The loaded dataset.
    """

    return datasets.MNIST(root=path,
                          train=False,
                          transform=transforms,
                          download=download)


collate_fn = None
get_dataset = prepare_mnist_dataset
