# Quantization and Sparsity exploration of DNNs

This project contains a Python library and test scripts to explore the behavior of DNN accuracy when quantization to inputs and weights are applied or when the sparsity of intermeditate results are artificially increased.


The aim of this project is to analyze the behavior of a model with Fake quantization layers in between.


## Setup & Prerequisities
The project comes as Python library, which can later be pubished to PyPI. The minimum supported Python version is Python 3.8

Currently, to install the package in an editable mode, run the following command, which installs the package. Pip will automatically install all dependencies. We still recommend to a create a virtual environment first.

```sh
python3.8 -m venv torch_exploration
pip install -e .
```

## Running a DNN model exploration
Our tool works entirely with workload description files, that contain all requried information. Just have a look at one of the sample yaml files in `./workloads`.

Key compoent is the problem definition at the beginning, which determines whether an exploration for increased sparsity or quantization should be started. 

Further down the file, you can adjust the parameters of the exploration algorithm and evaluation.

The according scripts to explore, retrain or calibrate are located in the `scripts` directory. 

To run a exploration that yields for example benefical sparsity thresholds for a sparsity problem, you can execute the following command:

```sh
python scripts/explore.py WORKLOAD_FILE [--skip-baseline] [--verbose] [--progress]
```

You can always call the given scripts in the scripts directory with the `--help` option to get all available command line options.

If a script has finished, it usually puts its results into a `results` directory together with a verbose log file. 


## Background

### How does a exploration run?

The exploration in this project uses the NSGA 2 algorithm to find suitable solutions.
Solutions are optimized for maximizing the resulting model accuracy and minimizing the second objective, which given by the problem.
As a first starting point it is advisable to check the `problems` directory for the problem definition.
When a problem is instanciated, first a model will be initalized. 
During initalization, a base model will be altered according to the problem. E.g. in a quantization problem, convolution layers will be replaced by counterparts that support fake quantization from the NVIDIA pytorch-quantization library.
Then a NSGA2 problem definition is build.
For optimization we use the Pymoo library. 
Each problem definition evaluates an altered model with a set of parameters (which are e.g. the bits for each individal layer in a quantization problem or the thresholds in a sparsity problem).
The problem function then returns the objectives, aka. the achieved accuracy and the e.g. the number of created sparse blocks in a sparsity problem or the total number of bits in a quantization problem.


The initial model must be calibrated first. This is done to determine the `amax` value (the maximum value within the provided calibration dataset). This value is required to later run the quantization.

In each iteration of the exploration a new list of `num_bits` is provided to update the `TensorQuantizer`s with. These `num_bits` are restrained to integer values only and are sampled from the range `\[num_bits_lower_limit, num_bits_upper_limit\]` defined in the `LayerwiseQuantizationProblem`.


### Quantization
Our project uses the `pytorch-quantization` package from NVIDIA. It can apply fake quantization to most of the common PyTorch layers. 

#### What is a Fake-Quantization?
A fake quantization takes an input tensor, quantizes it and immediately dequantizes it again.

In our case we use the `TensorQuantizer` from the [Nvidia pytorch-quantization](https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/userguide.html) module.
This module can be used like any other Pytorch module and performs quantization on the tensor passed to it.
A `TensorQuantizer` requires a `QuantDescriptor` that describes the quantization to be performed.
In our case we want to perform fake quantization and therefor set `fake_quant=True`. Additional parameters can be set as well.
One important attribute is the `num_bits` attribute which determines the number of bits to be used for quantization.

