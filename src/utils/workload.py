"""
This module contains the functionality to
load and handle workload .yaml files.

Raises:
    FileNotFoundError:
        When the workload file could not be found.
    ValueError:
        When the dataset setting is not present in the workload yaml.
    ValueError:
        When the model setting is not present in the workload yaml.
    ValueError:
        When no exploration setting is present in the workload yaml.
    ValueError:
        When no type is specified in the model setting inside the workload yaml.
"""
import os
import yaml


class Workload:
    """
    The workload class represents a workload loaded from a yaml file
    containing all relevant settings for the project.
    """

    def __init__(self, filename) -> None:
        self.filename = filename

        if not os.path.exists(filename):
            raise FileNotFoundError(f"Workload yaml file not found at {filename}")

        with open(filename, "r") as stream:
            self.yaml_data = yaml.safe_load(stream)["workload"]

    def __getitem__(self, item):
        if item in self.yaml_data:
            return self.yaml_data[item]
        return 0

    def get_dataset_settings(self):
        """Provides the dataset settings.

        Raises:
            ValueError:
                When the dataset setting is not present in the workload yaml.

        Returns:
            dict: The dict containing the dataset settings.
        """

        if "dataset" not in self.yaml_data:
            raise ValueError("Dataset settings not found in workload yaml")

        return self.yaml_data["dataset"]

    def get_model_settings(self):
        """Provides the model settings.

        Raises:
            ValueError:
                When the model setting is not present in the workload yaml.

        Returns:
            dict: The dict containing the model settings.
        """

        if "model" not in self.yaml_data:
            raise ValueError("Model settings not found in workload yaml")

        return self.yaml_data["model"]

    def get_nsga_settings(self):
        """Provides the nsga exploration settings.

        Raises:
            ValueError:
                When no exploration setting is present in the workload yaml.

        Returns:
            dict: The dict containing the nsga exploration settings.
        """

        if "nsga" not in self.yaml_data:
            raise ValueError("No setting found for exploration")

        return self.yaml_data["nsga"]

    def get_model_name(self):
        """Provides the module name/type.

        Raises:
            ValueError:
                When no type is specified in the model setting inside the workload yaml.

        Returns:
            dict: The name/type of the module.
        """

        model_settings = self.get_model_settings()

        if "type" not in model_settings:
            raise ValueError("Type not found in model settings")

        return self.yaml_data["model.type"]

    def get(self, item, default=None):
        """Gets the item from the workload.

        Args:
            item (str): The wanted item.
            default (any, optional):
                The default to return if the wanted item was not found.
                Defaults to None.

        Returns:
            The item if it was found in the yaml data, the default otherwise.
        """

        return self.yaml_data.get(item, default)
