from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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
    valid_dataset = prepare_mnist_dataset("mnist_data", download)

    return DataLoader(dataset=valid_dataset,
                      batch_size=BATCH_SIZE,
                      shuffle=False)


def prepare_mnist_dataset(path, download=False, **kwargs):
    """Load a mnist dataset.

    Args:
        path (str): Root directory for the mnist images.
        transforms (callable): A function/transform that takes input sample and its target as entry
        and returns a transformed version.
        download (bool): Wether to download the mnist dataset.

    Returns:
        datasets: The loaded dataset.
    """
    transf = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor()])

    return datasets.MNIST(root=path,
                          train=False,
                          transform=transf,
                          download=download)


collate_fn = None
get_validation_dataset = prepare_mnist_dataset
get_train_dataset = None
