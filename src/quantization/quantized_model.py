import torch
import torch.utils.data
import numpy as np

from tqdm import tqdm
from torch import nn

from torch.utils.data import DataLoader

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import quant_modules

from src.exploration.weighting_functions import bits_weighted_linear



class QuantizedModel():
    def __init__(self, 
                 model: nn.Module, 
                 device: torch.device, 
                 weighting_function: callable = bits_weighted_linear,
                 verbose=False
                 ) -> None:
        super().__init__()

        self.model = model
        self._bit_widths = {}
        self.device = device
        self.verbose = verbose
        self.weighting_function = weighting_function

        # supposingly this is not going to change
        self.quantizer_modules = []
        for _, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                self.quantizer_modules.append(module)
            

    @property
    def bit_widths(self):
        return self._bit_widths

    @bit_widths.setter
    def bit_widths(self, new_bit_widths):
        assert isinstance(new_bit_widths, list) or isinstance(new_bit_widths, np.ndarray), "bit_width have to be a list or ndarray"
        assert len(new_bit_widths) == len(self.quantizer_modules), "bit_width list has to match the amount of quantization layers"

        # Update Model ...
        for i, module in enumerate(self.quantizer_modules):
            module.num_bits = new_bit_widths[i]

        self._bit_widths = new_bit_widths

    def get_bit_weighted(self) -> int:
        return self.weighting_function(self.quantizer_modules)

    def enable_quantization(self):
        [module.enable_quant() for module in self.quantizer_modules]

    def disable_quantization(self):
        [module.disable_quant() for module in self.quantizer_modules]

    # CALIBRATION PART
    def load_calibration(self, filename: str):
        self.model.load_state_dict(torch.load(filename, map_location=self.device))
        
    def save_calibration(self, filename: str):
        torch.save(self.model.state_dict(), filename)
        
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
        for module in self.quantizer_modules:
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
        for module in self.quantizer_modules:
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

        # Collect amax statistics
        for module in self.quantizer_modules:
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)

