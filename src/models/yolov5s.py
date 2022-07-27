import torch
import sys

# sys.exit("YoloV5 is not yet traceable with this implementation of the sparse exploration.")

# The following code is needed to prevent package conflicts
# This is referenced in this issue:
# https://github.com/pytorch/hub/issues/243

CONFLIC_MODULES = ["models", "data", "utils"]


def _remove_modules(module_names: list) -> dict:

    removed_modules = {}

    for module_name in module_names:

        if module_name in sys.modules:
            removed_modules[module_name] = sys.modules.pop(module_name)

    return removed_modules


def _add_modules(modules: dict) -> None:

    for module_name, module in modules.items():
        sys.modules[module_name] = module


restore_modules = _remove_modules(CONFLIC_MODULES)

model = torch.hub.load("ultralytics/yolov5", "yolov5s", verbose=False)

_remove_modules(CONFLIC_MODULES)
_add_modules(restore_modules)
