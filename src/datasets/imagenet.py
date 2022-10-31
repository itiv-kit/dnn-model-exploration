import json

from torchvision import datasets




def get_imagenet_label_map(json_file):
    """Load the label map from the provided json_file.

    Args:
        json_file (str): The path to the label json file.

    Returns:
        dict: The loaded label map.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    ret_dict = {}
    
    for k,v in data.items():
        ret_dict.update({v[0] : int(k)})
        
    return ret_dict


def prepare_imagenet_dataset(path, transforms, **kwargs):
    """Load an imagenet dataset.

    Args:
        path (str): Root directory for the images.
        transforms (callable): A function/transform that takes input sample and its target as entry
        and returns a transformed version.

    Returns:
        datasets: The loaded dataset.
    """
    return datasets.ImageFolder(path, transforms)

collate_fn = None
get_dataset = prepare_imagenet_dataset