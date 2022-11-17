import numpy as np

def bits_weighted_linear(quantizer_modules, *args) -> int:
    bit_widths = [qm.num_bits for qm in quantizer_modules]
    return np.sum(bit_widths)

def bits_weighted_per_layer(quantizer_modules, quantizer_names) -> int:
    total = 0

    for module, name in zip(quantizer_modules, quantizer_names):
        bits = module.num_bits
        if 'weight' in name:
            total += 0.5 * bits
        elif 'input' in name:
            total += 5.0 * bits
        else:
            total += 1.0 * bits

    return total
