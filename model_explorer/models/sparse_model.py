import torch
import copy
import functools

from torch import nn

from model_explorer.models.sparse_convolution import SparseConv2d
from model_explorer.models.custom_model import CustomModel
from model_explorer.utils.logger import logger


class SparseModel(CustomModel):
    """The base model for our custom sparse models.
    """

    def __init__(self, model: nn.Module, block_size: int, device: torch.device, verbose: bool = False) -> None:
        """Initilizes a sparse model with the provided arguments.
        """
        super().__init__(model, device, verbose)

        self._thresholds = {}
        self._block_size = block_size
        self.sparse_modules = []
        self.sparse_module_names = []
        self._create_sparse_model()

    @property
    def thresholds(self):
        return self._thresholds

    @thresholds.setter
    def thresholds(self, new_thresholds: list):
        # TODO: Update model
        assert len(new_thresholds) == self.get_explorable_parameter_count()
        for i, module in enumerate(self.sparse_modules):
            module.threshold = new_thresholds[i]

        self._thresholds = new_thresholds

    def get_explorable_parameter_count(self) -> int:
        return len(self.sparse_modules)

    def get_total_created_sparse_blocks(self) -> int:
        return sum([module.sparse_created for module in self.sparse_modules])

    def reset_model_stats(self):
        [module.reset_stats() for module in self.sparse_modules]

    def _create_sparse_model(self):
        self.sparse_model = copy.deepcopy(self.model)

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                self.thresholds[name] = 0.0
                sparse_conv = SparseConv2d(module, self._block_size)
                self.sparse_modules.append(sparse_conv)
                self.sparse_module_names.append(name)
                # Replace actual conv2d with sparse_conv2d
                setattr(self.sparse_model, name, sparse_conv)

    # def __str__(self) -> str:
    #     return "Sparse Model, with {} replaced nodes".format(
    #         len(self.sparse_nodes))

    # def __repr__(self) -> str:
    #     return "Sparse Model, with {} replaced Nodes:\n\t{}".\
    #         format(len(self.sparse_nodes), ";\n\t".join(["{}, thres:{}, bs:{}".format(
    #             x['name'], x['threshold'], x['block_size'])
    #             for x in self.sparse_nodes]))
