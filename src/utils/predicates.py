"""
This module contains predicates to filter or search for specific modules in a torch model.
"""
import torch


def conv2d_predicate(module):
    """A predictae function that returns true if the provided module is a
    torch.nn.Conv2d module.

    Args:
        module (object): The object that should be tested on wether it is a torch.nn.Conv2d module.

    Returns:
        bool:  Wether the provided object is a torch.nn.Conv2d module.
    """
    return isinstance(module, torch.nn.Conv2d)
