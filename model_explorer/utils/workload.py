import os
import yaml
import collections.abc

from typing import Any


class Workload:
    """
    The workload class represents a workload loaded from a yaml file
    containing all relevant settings for the project.
    """

    def __init__(self, filename: str) -> None:
        self.filename = filename

        if not os.path.exists(filename):
            raise FileNotFoundError(f"Workload yaml file not found at {filename}")

        with open(filename, "r") as stream:
            self.yaml_data = yaml.safe_load(stream)["workload"]

        # check file for consistency
        assert "model" in self.yaml_data, "No model entry in yaml file given"
        assert "exploration" in self.yaml_data, "No exploration entry in yaml file given"
        assert "retraining" in self.yaml_data, "No retraining entry in yaml file given"
        assert "problem" in self.yaml_data, "No problem entry in yaml file given"


    def __getitem__(self, item: str):
        if item in self.yaml_data:
            return self.yaml_data[item]
        return None

    def merge(self, update_dict: dict):
        """Updates the loaded yaml configuration with other settings in the update dict
        """
        # https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
        def dict_update(d, u):
            for k, v in u.items():
                if isinstance(v, collections.abc.Mapping):
                    d[k] = dict_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        self.yaml_data = dict_update(self.yaml_data, update_dict)


    def get(self, item: str, default: Any = None):
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
