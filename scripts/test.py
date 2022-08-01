"""
This script is intended to be used as a test enviroment.
"""
import torch

from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer


# test the quantization modules

quant_desc = QuantDescriptor(
    num_bits=4,
    fake_quant=True,
    axis=(0),
    unsigned=False,
)

fake_quantizer = TensorQuantizer(quant_desc)

ts = torch.randn(1, 1, 3, 4)

# run calibration
fake_quantizer.enable_calib()
fake_quantizer.disable_quant()

fake_quantizer(ts)

fake_quantizer.load_calib_amax()
fake_quantizer.enable_quant()
fake_quantizer.disable_calib()


print(ts)
print(fake_quantizer._amax)

res = fake_quantizer(ts)

print(res)

# change num bits
fake_quantizer.num_bits = 2
print("Change num_bits to 2")
res = fake_quantizer(ts)

print(res)
