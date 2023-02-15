import numpy as np


def bits_weighted_linear(quantizer_modules: list, *args) -> int:
    """Weighting function that sums up the bits in a quantized model

    Args:
        quantizer_modules (list): List of model modules

    Returns:
        int: Sum of bits
    """
    bit_widths = [qm.num_bits for qm in quantizer_modules]
    return np.sum(bit_widths)


def bits_weighted_per_layer(quantizer_modules: list, quantizer_names: list) -> int:
    """Weighting function that sums up the bits in a quantized model, but puts
    differnt emphasis on the different modules, e.g., input quantizer or weight
    quantizer

    Args:
        quantizer_modules (list): List of model modules
        quantizer_names (list): List of module names

    Returns:
        int: weighted sum of bits
    """
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
