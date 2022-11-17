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
from src.utils.logger import logger
from src.utils.data_loader_generator import DataLoaderGenerator


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

        # Training things
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)
            

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

    # RETRAIN PART
    def retrain(self, train_dataloader_generator: DataLoaderGenerator,
                test_dataloader_generator: DataLoaderGenerator, 
                accuracy_function: callable,
                num_epochs=10, progress=False):
        # Run Training
        self.model.to(self.device)

        for epoch_idx in range(num_epochs):
            self.model.train()

            logger.info("Starting Epoch {} / {}".format(epoch_idx+1, num_epochs))
            
            if progress:
                pbar = tqdm(total=len(train_dataloader_generator), 
                            desc="Epoch {} / {}".format(epoch_idx+1, num_epochs),
                            position=1)
                
            running_loss = 0.0
            train_dataloader = train_dataloader_generator.get_dataloader()

            for image, target, *_ in train_dataloader:
                image, target = image.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                
                with torch.set_grad_enabled(mode=True):
                    output = self.model(image)
                    loss = self.criterion(output, target)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item() * output.size(0)

                    if progress:
                        pbar.update(output.size(0))

            self.lr_scheduler.step()

            if progress:
                pbar.close()

            epoch_loss = running_loss / len(train_dataloader_generator)
            logger.info("Ran Epoch {} / {} with loss of: {}".format(epoch_idx+1, num_epochs, epoch_loss))

            self.model.eval()
            # FIXME! 
            test_dataloader = test_dataloader_generator.get_dataloader()
            acc = accuracy_function(self.model, test_dataloader, progress, title="Eval {} / {}".
                                    format(epoch_idx+1, num_epochs))
            logger.info("Inference Accuracy after Epoch {}: {}".format(epoch_idx+1, acc))
        
    # LOADING and STORING
    def load_parameters(self, filename: str):
        self.model.load_state_dict(torch.load(filename, map_location=self.device))
        
    def save_parameters(self, filename: str):
        torch.save(self.model.state_dict(), filename)
        
    # CALIBRATION PART
    def run_calibration(self, dataloader: DataLoader, progress=True, calib_method='histogram', **kwargs):
        assert calib_method in ['max', 'histogram'], "method has to be either max or histogram"

        quant_desc_input = QuantDescriptor(calib_method=calib_method)
        quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantLSTMCell.set_default_quant_desc_input(quant_desc_input)
        
        self._collect_stats(dataloader=dataloader, progress=progress, kwargs=kwargs)

    def _collect_stats(self, dataloader, progress, kwargs):
        self.model.to(self.device)
        
        # Enable Calibrators
        for module in self.quantizer_modules:
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

        # Run the dataset ...
        for data, *_ in tqdm(dataloader, desc="Calibrating", disable=progress):
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

