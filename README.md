# Quantization-aware Robustness Exploration of DNNs

This project contains a Python library and test scripts to explore the behavior of a DNN model accuracy when quantization to inputs and weights is applied under various weather conditions.

The aim of this project is to analyze the behavior of a model in which layers are replaced with counterparts that model quantized inference.

This project can also be extended to enable other DNN behavior, if this is preferred. Have a look into the quick guide down below. Otherwise, feel free to open a question or even a pull request.


## Setup & Prerequisities
The project comes as Python library, which can later be published to PyPI. We recommend using Python3.9, however, 3.8 or 3.10 are likely to work as well.

Currently, to install the package in an editable mode, run the following command, which installs the package. Pip will automatically install all dependencies. We still recommend to a create a virtual environment first.

```sh
python3.9 -m venv torch_exploration
pip install -e .
```

Second, if you need them, checkout the submodules for YoloP, DeepLabV3 and Timeloop
```sh
git submodule update --init
```

Now you should be set for a model exploration.


## Running a DNN model exploration
Our tool works entirely with workload description files, that contain all required information. Just have a look at one of the sample yaml files in `./workloads`.
Key component is the problem definition at the beginning.
Currently, two options are available: `quantization_problem` for an exploration of mixed-precision inference and `energy_aware_quant_problem` for a mixed-precision execution with focus on DRAM memory access savings.
Further down the file, you can adjust the parameters of the exploration algorithm and evaluation.

The according scripts to explore, retrain or calibrate are located in the `scripts` directory. After a script has successfully ran, results are always stored in a `results` folder. Exploration results are always stored as pickle files containing all information gathered during exploration for later evaluation (these files can easily grow to 10 GB). 

To run a exploration that yields for example beneficial bit-width combinations for a quantization problem, you can execute the following command:

```sh
python scripts/explore.py WORKLOAD_FILE [--skip-baseline] [--verbose] [--progress]
```

You can always call the given scripts in the scripts directory with the `--help` option to get all available command line options.


## Evaluation of results
Some handy scripts are available in the `evaluation_scripts` folder.

- `dram_accesses.ipynb` reads csv files generated by `scripts/energy_analysis.py` to get the total number of DRAM accesses
- `visualize_individuals.ipynb` has some functions to plot all individuals found during exploration and exports data to TikZ for scientific papers
- `analyse_weather.ipynb` a workbook to generate plots and insights on how model compression goes along with inputs that are corrupted by, e.g., rain, fog or snow.


## Number converter architecture
Our scalable hardware architecture to enable mixed-precision inference can be found in `./number_converter_hw`.
`src` contains the Verilog implementation of the components, while test scripts are located in `testbenches`.
We used Vivado 2022.1 to map the architecture to the target FPGA and Modelsim for simulation. 
However, you should be able to use any other design tools.


## Extension of the project
If you like to add you own problem definition or add new models, feel free to extend this project!

### Adding a new model
Check out the files in `model_explorer/workloads`.
Each always have to describe a `model`, which might simply be taken from torchvision or can be defined in place; and an `accuracy_function` which is executed when the model accuracy impact is computed.
There is a selection of commonly-used accuracy function available in `model_explorer/accuracy_functions`.
However, you can create your own if you need to.
Be aware that you might need to add the corresponding dataset as well.
To do so, have a look into the `datasets` directory.

### Adding a new problem description
For that see the already provided problems in `model_explorer/problems`.
Each problem description always has to define the following five things:
1. `prepare_exploration_function`: this function is called when the problem is constructed. Here you may initialize the model and everything that is needed to run an exploration.
2. `repair_method`: This mainly depends on PyMoo. The repair method is important when dealing with exploration parameters that are integers rather than continuous numbers.
3. `sampling_method`: Similar to `repair_method`
4. `init_function`: This is the function that builds the model and returns the model that matches the exploration problem
5. `update_params_function`: This function updates the model with a new set of exploration parameters.


## Background

### How does a exploration run?

The exploration in this project uses the NSGA 2 algorithm to find suitable solutions.
Solutions are optimized for maximizing the resulting model accuracy and minimizing the second objective, which given by the problem.
As a first starting point it is advisable to check the `problems` directory for the problem definition.
When a problem is instantiated, first a model will be initialized. 
During initialization, a base model will be altered according to the problem. E.g. in a quantization problem, convolution layers will be replaced by counterparts that support fake quantization from the NVIDIA pytorch-quantization library.
Then a NSGA2 problem definition is build.
For optimization we use the Pymoo library. 
Each problem definition evaluates an altered model with a set of parameters (which are e.g. the bits for each individual layer in a quantization problem).


The initial model must be calibrated first. This is done to determine the `amax` value (the maximum value within the provided calibration dataset). This value is required to later run the quantization.

In each iteration of the exploration a new list of `num_bits` is provided to update the `TensorQuantizer`s with. These `num_bits` are restrained to integer values only and are sampled from the range `[num_bits_lower_limit, num_bits_upper_limit]` defined in the `LayerwiseQuantizationProblem`.


### Quantization
Our project uses the `pytorch-quantization` package from NVIDIA. It can apply fake quantization to most of the common PyTorch layers. 

#### What is a Fake-Quantization?
A fake quantization takes an input tensor, quantizes it and immediately dequantizes it again.

In our case we use the `TensorQuantizer` from the [Nvidia pytorch-quantization](https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/userguide.html) module.
This module can be used like any other Pytorch module and performs quantization on the tensor passed to it.
A `TensorQuantizer` requires a `QuantDescriptor` that describes the quantization to be performed.
In our case we want to perform fake quantization and therefor set `fake_quant=True`. Additional parameters can be set as well.
One important attribute is the `num_bits` attribute which determines the number of bits to be used for quantization.

