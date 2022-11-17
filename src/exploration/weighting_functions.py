import numpy as np

def bits_weighted_linear(quantizer_modules) -> int:
    bit_widths = [qm.num_bits for qm in quantizer_modules]
    return np.sum(bit_widths)

