from torchvision import datasets, transforms
import os


def prepare_imagenet_c_dataset(path: str, kind: str, **kwargs):
    """Load a imagenet corrupted dataset.

    Args:
        path (str): Root directory for the images.
        kind (str): Either webdataset or imagefolder for the according format
        and returns a transformed version.

    Returns:
        datasets: The loaded dataset.
    """

    assert kind in ['webdataset', 'imagefolder'], "Kind has to be either webdataset or imagefolder"

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(), normalize
    ])

    severity = kwargs.get('severity', 1)
    condition = kwargs.get('condition', 'fog')
    assert severity in range(1, 6)
    assert condition in ['fog', 'frost', 'snow', 'brightness']

    imagenet_path = os.path.join(path, condition, str(severity))

    if kind == 'imagefolder':
        return datasets.ImageFolder(imagenet_path, transf)
    else:
        raise NotImplementedError("For Imagenet-C only image folders are supported")


dataset_creator = prepare_imagenet_c_dataset
collate_fn = None
