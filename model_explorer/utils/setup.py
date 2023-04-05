"""
Contains the setup logic to dynamically load the required model, dataset and utils modules.
"""
import importlib
import torch
from model_explorer.utils.data_loader_generator import DataLoaderGenerator


MODEL_FOLDER = "..workloads"
ACCURACY_FUNCTIONS_FOLDER = "..accuracy_functions"
TRANSFORMS_FOLDER = "..transforms"
DATASETS_FOLDER = "..datasets"
PROBLEMS_FOLDER = "..problems"


def setup_workload(model_settings: dict) -> list:
    """This function sets up the model and returns the model as well as the accuracy function.

    Args:
        model_settings (dict): The dict containing the settings for the specified model.

    Returns:
        tuple: A tuple containing the loaded model and its accuracy function.
    """

    # handle models as plugins inside `/models` and load them dynamically
    # this requires `model` to be defined in the model module
    model = importlib.import_module(
        f"{MODEL_FOLDER}.{model_settings['type']}", package=__package__
    ).model

    accuracy_function = importlib.import_module(
        f"{ACCURACY_FUNCTIONS_FOLDER}.{model_settings['accuracy_function']}",
        package=__package__,
    ).accuracy_function

    return model, accuracy_function


def get_prepare_exploration_function(problem_name: str) -> tuple:
    """This function returns the preparation function which is defined in the
    problem file together the repair and sampling method. Functions are loaded
    form the according problem description file

    Args:
        problem_name (str): name of the problem definition, matches the file in problems

    Returns:
        list: length = 3, aforementioned items
    """
    prepare_exploration_function = importlib.import_module(
        f"{PROBLEMS_FOLDER}.{problem_name}", package=__package__
    ).prepare_exploration_function
    repair_method = importlib.import_module(
        f"{PROBLEMS_FOLDER}.{problem_name}", package=__package__
    ).repair_method
    sampling_method = importlib.import_module(
        f"{PROBLEMS_FOLDER}.{problem_name}", package=__package__
    ).sampling_method
    return prepare_exploration_function, repair_method, sampling_method


def get_model_init_function(problem_name: str) -> callable:
    """Load the model initalization function from the problem definition file

    Args:
        problem_name (str): problem name, matching the filename in problems dir

    Returns:
        callable: initialization function reference
    """
    init_func = importlib.import_module(
        f"{PROBLEMS_FOLDER}.{problem_name}", package=__package__
    ).init_function
    return init_func


def get_model_update_function(problem_name: str) -> callable:
    """Load the update function for a given model that can be used to update the explorable parameters

    Args:
        problem_name (str): problem name, matching the filename in the problems dir

    Returns:
        callable: Update function reference
    """
    update_func = importlib.import_module(
        f"{PROBLEMS_FOLDER}.{problem_name}", package=__package__
    ).update_params_function
    return update_func


def setup_dataset(dataset_settings: dict) -> list:
    """This function sets up the dataset and returns the dataset.

    Args:
        dataset_settings (dict): The dict containing the settings for the specified dataset.

    Returns:
        list: Dataset and collate function
    """

    # handle datasets as plugins inside `/datasets` and load them dynamically
    # this requires `get_dataset` function to be defined in the dataset module

    dataset_module = importlib.import_module(
        f"{DATASETS_FOLDER}.{dataset_settings['type']}", package=__package__
    )

    dataset_creator = dataset_module.dataset_creator
    dataset = dataset_creator(**dataset_settings)

    collate_fn = dataset_module.collate_fn

    return dataset, collate_fn


def build_dataloader_generators(settings_dict: dict) -> dict:
    """Helper function to generate all available dataloader generators

    Args:
        settings_dict (dict): sub-dictionary taken from the workload file

    Returns:
        dict: collection of generated dataloader generators
    """
    ret_dict = {}
    for dataset_name, dataset_settings in settings_dict.items():
        dataset, collate = setup_dataset(dataset_settings)
        loader_generator = DataLoaderGenerator(dataset,
                                               collate,
                                               items=dataset_settings.get('total_samples', None),
                                               batch_size=dataset_settings['batch_size'],
                                               limit=dataset_settings.get('sample_limit', None),
                                               randomize=dataset_settings.get('randomize', False))
        ret_dict[dataset_name] = loader_generator

    return ret_dict


def setup_torch_device() -> torch.device:
    """Helper to setup torch device, checks if gpu is available

    Returns:
        torch.device: available best device choice
    """
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'

    return torch.device(device_str)
