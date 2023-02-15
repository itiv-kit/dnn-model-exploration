import torch
import functools

from torch import nn

from model_explorer.models.sparse_convolution import SparseConv2d
from model_explorer.models.custom_model import CustomModel


class SparseModel(CustomModel):
    """A sparse model automatically replaces all convolutions with SparseConv2d
    modules and applies a set of thresholds and the given block size
    """

    def __init__(self, base_model: nn.Module, block_size: list, device: torch.device,
                 collect_sparsity_details: bool = True, verbose: bool = False):
        super().__init__(base_model, device, verbose)

        self._thresholds = {}
        self._collect_sparsity_details = collect_sparsity_details
        # For now, block size cannot be changed dynamically
        assert len(block_size) == 2, "block size parameter has to be a list with 2 elements: width and height"
        self._block_size = block_size
        self._create_sparse_model()

    @property
    def thresholds(self):
        return self._thresholds

    @thresholds.setter
    def thresholds(self, new_thresholds: list):
        assert len(new_thresholds) == self.get_explorable_parameter_count()
        for i, module in enumerate(self.explorable_modules):
            module.threshold = new_thresholds[i]

        self._thresholds = new_thresholds

    def get_total_created_sparse_blocks(self) -> int:
        return sum([module.sparse_created for module in self.explorable_modules])

    def get_total_present_sparse_blocks(self) -> int:
        return sum([module.sparse_present for module in self.explorable_modules])

    def reset_model_stats(self):
        [module.reset_stats() for module in self.explorable_modules]

    def _create_sparse_model(self):
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Conv2d):
                self.thresholds[name] = 0.0
                sparse_conv = SparseConv2d(module, self._block_size, self._collect_sparsity_details)
                self.explorable_modules.append(sparse_conv)
                self.explorable_module_names.append(name)

                # Replace actual conv2d with sparse_conv2d
                # FIXME: is this save for all networks?
                module_name = name.split('.')[-1]
                module_path = name.split('.')[:-1]
                module_parent = functools.reduce(getattr, [self.base_model] + module_path)
                setattr(module_parent, module_name, sparse_conv)
