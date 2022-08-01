"""
This module contains quantization strategies for a pytorch model.
The doc for the library used can be found here: 
https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/userguide.html
"""
import logging
from src.utils.logger import logger
import torch
from torch import nn

from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer

# disable verbose logging from pytorch_quantization module
logging.getLogger("absl").disabled = True

# for all modules matching the predicate set a FakeQuant module in the forward hook
# run model with test data to get FakeQuantParams (deaktivate Quantization for that)
# aktivate quant again

# On ResNet50 w/ ImageNet: 1024 samples (2 batches of 512) should be sufficient to estimate the distribution of activations
# Src: https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/tutorials/quant_resnet50.html#calibration


class QuantizedActivationsModel(nn.Module):
    """
    Places and manages fake quantizations in the forward hook
    of selected modules in a provided model.
    """

    def __init__(
        self, model: nn.Module, predicate: callable, init_n_bits=8, unsigned=False
    ) -> None:
        super().__init__()

        self._model = model

        self._predicate = predicate
        self._init_n_bits = init_n_bits
        self._unsigned = unsigned

        self.activation_fake_quantizers = {}

        # since we can only register the fake quantization modules in the forward hook
        # we have to add an additional one before our model to quantize the input
        fake_quantizer = self._getTensorQuantizer("input")
        self.quant_model = nn.Sequential(fake_quantizer, self._model)

        self._quantize_layer_activations()

    def update_layer_n_bits(self, new_n_bits) -> None:
        """Update the bit resolution for each quantized layer.

        Args:
            new_n_bits (list): A list with the number of bits for each layer.
        """

        assert (
            len(new_n_bits) == self.get_n_quantizers()
        ), "Number of new quantization bit sizes has to equal number of quaantization layers."

        for i, fake_quantizer in enumerate(self.activation_fake_quantizers.values()):

            if not new_n_bits[i]:
                # n_bit is zero meaning we dont want to quantize this layer
                fake_quantizer.disable_quant()

            else:
                fake_quantizer.enable_quant()
                fake_quantizer.num_bits = new_n_bits[i]

    def calibrate(self, calibration_data_loader) -> None:
        """Calibrates the quantizers by running inference on the model
        and collecting data on the activations.

        Args:
            calibration_data_loader (DataLoader):
                The data loader with the samples to calibrate the model.
                Around 1024 should be sufficient.
        """

        logger.info("Starting model calibration.")

        # enable calibration on full precision
        for fake_quantizer in self.activation_fake_quantizers.values():
            fake_quantizer.enable_calib()
            fake_quantizer.disable_quant()

        # TODO Generalize!
        # tqdm bar and logger that we are calibrating
        for (im, targets, paths, shapes) in calibration_data_loader:

            if torch.cuda.is_available():
                im = im.cuda()
            self.quant_model(im)

        for fake_quantizer in self.activation_fake_quantizers.values():
            fake_quantizer.load_calib_amax()
            fake_quantizer.enable_quant()
            fake_quantizer.disable_calib()

        logger.info("Finished model calibration.")

        if torch.cuda.is_available():
            self.quant_model.cuda()

    def _getTensorQuantizer(self, name_prefix):

        quant_desc = QuantDescriptor(
            num_bits=self._init_n_bits,
            fake_quant=True,
            axis=None,  # (0), TODO: figure out what we need here
            unsigned=self._unsigned,
        )

        fake_quantizer = TensorQuantizer(quant_desc)
        name = f"{name_prefix}_fake_quantizer"
        fake_quantizer.name = name

        # register the quantizer in our dict to keep track of all existing ones
        self.activation_fake_quantizers[name] = fake_quantizer

        return fake_quantizer

    def _quantize_layer_activations(self) -> None:

        for name, module in self._model.named_modules():
            if self._predicate(module):

                fake_quantizer = self._getTensorQuantizer(name)

                module.register_forward_hook(
                    lambda module, inp, out, quant=fake_quantizer: self._fake_quantization_hook(
                        module, inp, out, quant
                    )
                )

        logger.info(
            f"{len(self.activation_fake_quantizers)} TensorQuantizers have been registered."
        )

    def _fake_quantization_hook(
        self,
        module: nn.Module,
        input: torch.Tensor,
        output: torch.Tensor,
        fake_quantizer: TensorQuantizer,
    ) -> torch.Tensor:
        return fake_quantizer(output)

    def get_n_quantizers(self) -> int:
        """Get the number of quantizers registered in the model.

        Returns:
            int: Number of quantizers registered in the model.
        """
        return len(self.activation_fake_quantizers)

    def forward(self, x: torch.Tensor):
        return self.quant_model(x)
