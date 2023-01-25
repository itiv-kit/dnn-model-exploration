import torch
import functools
import numpy as np

from tqdm import tqdm
from torch import nn as torch_nn

from torch.utils.data import DataLoader

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization import tensor_quant
from pytorch_quantization.tensor_quant import QuantDescriptor

from model_explorer.exploration.weighting_functions import bits_weighted_linear
from model_explorer.models.custom_model import CustomModel


class QuantizedModel(CustomModel):

    def __init__(self,
                 model: torch_nn.Module,
                 device: torch.device,
                 weighting_function: callable = bits_weighted_linear,
                 verbose=False) -> None:
        super().__init__(model, device, verbose)

        self._bit_widths = {}
        self.weighting_function = weighting_function

        # supposingly this is not going to change
        self.explorable_modules = []
        self.explorable_module_names = []
        self._create_quant_model()

    @property
    def bit_widths(self):
        return self._bit_widths

    @bit_widths.setter
    def bit_widths(self, new_bit_widths):
        assert isinstance(new_bit_widths, list) or isinstance(
            new_bit_widths,
            np.ndarray), "bit_width have to be a list or ndarray"
        assert len(new_bit_widths) == len(
            self.explorable_modules
        ), "bit_width list has to match the amount of quantization layers"

        # Update Model ...
        for i, module in enumerate(self.explorable_modules):
            module.num_bits = new_bit_widths[i]

        self._bit_widths = new_bit_widths

    def get_explorable_parameter_count(self) -> int:
        return len(self.explorable_modules)

    def get_bit_weighted(self) -> int:
        return self.weighting_function(self.explorable_modules,
                                       self.explorable_module_names)

    def enable_quantization(self):
        [module.enable_quant() for module in self.explorable_modules]

    def disable_quantization(self):
        [module.disable_quant() for module in self.explorable_modules]

    def _create_quant_model(self) -> None:
        for name, module in self.base_model.named_modules():
            if isinstance(module, torch_nn.modules.conv.Conv2d):
                # quant_conv = quant_nn.QuantConv2d(**module.__dict__)
                quant_conv = quant_nn.QuantConv2d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    bias=module.bias,
                    padding_mode=module.padding_mode,
                    quant_desc_input=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR,
                    quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
                )

                # FIXME: is this save for all networks?
                module_name = name.split('.')[-1]
                module_path = name.split('.')[:-1]
                module_parent = functools.reduce(getattr, [self.base_model] + module_path)
                setattr(module_parent, module_name, quant_conv)

        for name, module in self.base_model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                self.explorable_module_names.append(name)
                self.explorable_modules.append(module)


    # CALIBRATION PART
    def run_calibration(self,
                        dataloader: DataLoader,
                        progress=True,
                        calib_method='histogram',
                        **kwargs):
        assert calib_method in ['max', 'histogram'
                                ], "method has to be either max or histogram"

        quant_desc_input = QuantDescriptor(calib_method=calib_method)
        quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantLSTMCell.set_default_quant_desc_input(quant_desc_input)

        self._collect_stats(dataloader=dataloader,
                            progress=progress,
                            kwargs=kwargs)

    def _collect_stats(self, dataloader, progress, kwargs):
        self.base_model.to(self.device)

        # Enable Calibrators
        for module in self.explorable_modules:
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

        # Run the dataset ...
        for data, *_ in tqdm(dataloader,
                             desc="Calibrating",
                             disable=not progress):
            # no need for actual accuracy function ...
            self.base_model(data.to(self.device))

        # Disable Calibrators
        for module in self.explorable_modules:
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

        # Collect amax statistics
        for module in self.explorable_modules:
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
