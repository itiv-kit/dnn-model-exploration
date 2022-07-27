"""
Contains the setup logic to dynamicly load the required model, dataset and utils modules.
"""
import importlib
from src.utils.logger import logger
from .workload import Workload


MODEL_FOLDER = "..models"
ACCURACY_FUNCTIONS_FOLDER = "..accuracy_functions"
TRANSFORMS_FOLDER = "..transforms"
DATASETS_FOLDER = "..datasets"


def setup_model(model_settings: dict) -> list:
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

    transforms = importlib.import_module(
        f"{TRANSFORMS_FOLDER}.{model_settings['transforms']}", package=__package__
    ).transforms

    return model, accuracy_function, transforms


def setup_dataset(dataset_settings: dict, transforms) -> list:
    """This function sets up the dataset and returns the dataset.

    Args:
        dataset_settings (dict): The dict containing the settings for the specified dataset.

    Returns:
        Dataset: The loaded dataset.
    """

    # handle datasets as plugins inside `/datasets` and load them dynamically
    # this requires `get_dataset` function to be defined in the dataset module

    dataset_module = importlib.import_module(
        f"{DATASETS_FOLDER}.{dataset_settings['type']}", package=__package__
    )

    get_dataset = dataset_module.get_dataset
    collate_fn = dataset_module.collate_fn

    dataset = get_dataset(**dataset_settings, transforms=transforms)

    return dataset, collate_fn


def setup(workload: Workload) -> list:
    """Loads the model, accuracy_function, dataset, collate_fn to be used.

    Args:
        workload (Workload): The workload containing the setup setting.

    Returns:
        list: Returns a list consisting of the model, accuracy funtion,
        dataset and collate function for the dataloader.
    """

    model_settings = workload.get_model_settings()
    dataset_settings = workload.get_dataset_settings()

    model, accuracy_function, transforms = setup_model(model_settings)
    dataset, collate_fn = setup_dataset(dataset_settings, transforms)

    logger.info("Setup finished.")

    return model, accuracy_function, dataset, collate_fn
