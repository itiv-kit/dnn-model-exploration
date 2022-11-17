import json

from torchvision import datasets, transforms
import webdataset as wds


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



def prepare_imagenet_dataset(path, kind, **kwargs):
    """Load an imagenet dataset.

    Args:
        path (str): Root directory for the images.
        and returns a transformed version.

    Returns:
        datasets: The loaded dataset.
    """

    def identity(d):
        return d

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    transf = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                normalize])
        
    if kind == 'imagefolder':
        return datasets.ImageFolder(path, transf)
    elif kind == 'webdataset':
        dataset_load = (
            wds.WebDataset(path, shardshuffle=True)
            .decode("pil")
            .to_tuple("input.jpeg", "target.cls")
            .shuffle(10000)
            .map_tuple(transf, identity)
        )
        return dataset_load
    


dataset_creator = prepare_imagenet_dataset
collate_fn = None

