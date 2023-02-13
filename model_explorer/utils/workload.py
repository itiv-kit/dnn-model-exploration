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
import collections.abc


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

        # check file for consistency
        assert "model" in self.yaml_data, "No model entry in yaml file given"
        assert "exploration" in self.yaml_data, "No exploration entry in yaml file given"
        assert "retraining" in self.yaml_data, "No retraining entry in yaml file given"
        assert "problem" in self.yaml_data, "No problem entry in yaml file given"


    def __getitem__(self, item):
        if item in self.yaml_data:
            return self.yaml_data[item]
        return None

    def merge(self, update_dict):
        # https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
        def dict_update(d, u):
            for k, v in u.items():
                if isinstance(v, collections.abc.Mapping):
                    d[k] = dict_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        self.yaml_data = dict_update(self.yaml_data, update_dict)


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
