import torch
import torch.utils.data
import math

from tqdm import tqdm
from torch import nn

from torch.utils.data import DataLoader

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import quant_modules



class QuantizedModel():
    def __init__(self, model: nn.Module, device: torch.device, predicate: callable, verbose=False) -> None:
        super().__init__()

        self.model = model
        self._bit_widths = {}
        self.device = device
        self.predicate = predicate
        self.verbose = verbose

        #inject monkey patching code into the loaded model
        quant_modules.initialize()

    @property
    def bit_widths(self):
        return self._bit_widths

    @bit_widths.setter
    def bit_widths(self, new_bit_widths):
        assert isinstance(new_bit_widths, dict), "bit_width have to be a dict"

        # Update Model ...
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                assert name in new_bit_widths, "Layer {} not found in bit_widths dict".format(name)
                module.num_bits = new_bit_widths[name]

        self._bit_widths = new_bit_widths

    def disable_quantization(self):
        #disable injections again
        quant_modules.deactivate()

    # CALIBRATION PART
    def run_calibration(self, dataloader: DataLoader, calib_method='histogram', **kwargs):
        assert calib_method in ['max', 'histogram'], "method has to be either max or histogram"

        quant_desc_input = QuantDescriptor(calib_method=calib_method)
        quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantLSTMCell.set_default_quant_desc_input(quant_desc_input)
        
        self._collect_stats(dataloader=dataloader, kwargs=kwargs)

    def _collect_stats(self, dataloader, kwargs):
        self.model.to(self.device)
        
        # Enable Calibrators
        for _, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()

        # Run the dataset ...
        for data, *_ in tqdm(dataloader, desc="Calibrating"):
            # no need for actual accuracy function ...
            self.model(data.to(self.device))

        # Disable Calibrators
        for _, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()

        # Collect amax statistics
        for _, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax()
                    else:
                        module.load_calib_amax(**kwargs)

