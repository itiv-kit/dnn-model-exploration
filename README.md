# Quantization analysis

The aim of this project is to analyze the behavior of a model with Fake quantization layers in between.

## Introduction

### What is a Fake-Quantization?

A fake quantization takes an input tensor, quantizes it and immediately dequantizes it again.

In our case we use the `TensorQuantizer` from the [Nvidia pytorch-quantization](https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/userguide.html) module.
This module can be used like any other Pytorch module and performs quantization on the tensor passed to it.
A `TensorQuantizer` requires a `QuantDescriptor` that describes the quantization to be performed.
In our case we want to perform fake quantization and therefor set `fake_quant=True`. Additional parameters can be set as well.
One important attribute is the `num_bits` attribute which determines the number of bits to be used for quantization.

### How is the quantization applied?

We register `TensorQuantizer`s through two different methods.

1. We select modules of the given model through a predicate (see `utils/predicates.py`) and register a `TensorQuantizer` in this module forward hook call. Each module receives their own `TensorQuantizer`. These can also be accessed throght `activation_fake_quantizers` in `QuantizedActivationsModel`. This allows to alter them during the exploration.

    Each `TensorQuantizer` is called **after** the forward call of their respective module and receives the output of the module it is registered in.

2. Since we also want to quantize the input we also create a `nn.Sequential` and insert a `TensorQuantizer` followed by our model.

### How is the exploration run?

Currently we use the NSGA 2 algorithm to find suitable solutions.
Solutions are optimized for maximizing the resulting model accuracy and minimizing the total sum of `num_bits` over all used `TensorQuantizer`s.
This aims at achieving an accurate model with as little bit resolution as possible.
Additionally, a constraint can be given to the algorithm to restrict the minimal accuracy that is accepted.

To perform an exploration, multiple steps are needed:

0. Setup the model (`QuantizedActivationsModel`) and calibrate it.
1. Declare the problem to be minimized. (`LayerwiseQuantizationProblem`)
2. Declare the algorithm to be used. (NSGA 2)
3. Minimize the problem with the given algorithm.

The initial model must be calibrated first. This is done to determine the `amax` value (the maximum value within the provided calibration dataset). This value is required to later run the quantization.

In each iteration of the exploration a new list of `num_bits` is provided to update the `TensorQuantizer`s with. These `num_bits` are restrained to integer values only and are sampled from the range `\[num_bits_lower_limit, num_bits_upper_limit\]` defined in the `LayerwiseQuantizationProblem`.

## Setup

### Src files

To make the source files available run the following command from the project root.

```sh
pip install -e .
```

### Dependencies

The dependencies can be installed using conda and the provided `environment.yml`.
