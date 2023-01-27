import torch
import os
import importlib

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from torch import nn

from model_explorer.problems.custom_problem import CustomExplorationProblem
from model_explorer.utils.logger import logger
from model_explorer.models.quantized_model import QuantizedModel
from model_explorer.utils.data_loader_generator import DataLoaderGenerator


def update_quant_model_params(qmodel: QuantizedModel, bits: list):
    qmodel.bit_widths = bits


def init_quant_model(model: nn.Module, device: torch.device,
                     verbose: bool, **kwargs: dict) -> QuantizedModel:
    weighting_function_name = kwargs.get('bit_weighting_function')
    calibration_file = kwargs.get('calibration_file')

    weighting_function = getattr(
        importlib.import_module('model_explorer.exploration.weighting_functions'),
        weighting_function_name, None)
    assert weighting_function is not None and callable(
        weighting_function), "error loading weighting function"

    qmodel = QuantizedModel(model,
                            device,
                            weighting_function=weighting_function,
                            verbose=verbose)

    logger.debug("Added {} Quantizer modules to the model".format(
        len(qmodel.explorable_modules)))

    # Load the previously generated calibration file
    if not os.path.exists(calibration_file):
        logger.error("Calibtraion file not found")
        raise FileNotFoundError("Calibration file not found")

    logger.debug(f"Loading calibration file: {calibration_file}")
    qmodel.load_parameters(calibration_file)

    return qmodel


def prepare_quantization_problem(model: nn.Module, device: torch.device,
                                 dataloader_generator: DataLoaderGenerator,
                                 accuracy_function: callable, min_accuracy: float,
                                 verbose: bool, progress: bool,
                                 **kwargs: dict):
    num_bits_upper_limit = kwargs.get('num_bits_upper_limit')
    num_bits_lower_limit = kwargs.get('num_bits_lower_limit')

    qmodel = init_quant_model(model, device, verbose, kwargs)
    logger.info("Quantization problem and model initialized")

    return LayerwiseQuantizationProblem(
        qmodel=qmodel,
        dataloader_generator=dataloader_generator,
        accuracy_function=accuracy_function,
        min_accuracy=min_accuracy,
        progress=progress,
        num_bits_lower_limit=num_bits_lower_limit,
        num_bits_upper_limit=num_bits_upper_limit)


class LayerwiseQuantizationProblem(CustomExplorationProblem):
    """
    A pymoo problem defenition for the quantization exploration.
    """

    def __init__(
        self,
        qmodel: QuantizedModel,
        dataloader_generator: DataLoaderGenerator,
        accuracy_function: callable,
        min_accuracy: float,
        progress: bool,
        num_bits_upper_limit: int,
        num_bits_lower_limit: int,
        **kwargs,
    ):
        """Inits a quantization exploration problem.
        """
        super().__init__(
            model=qmodel,
            accuracy_function=accuracy_function,
            progress=progress,
            min_accuracy=min_accuracy,
            n_var=qmodel.get_explorable_parameter_count(),
            n_constr=1,  # accuracy constraint
            n_obj=2,  # accuracy and low bit num
            xl=num_bits_lower_limit,
            xu=num_bits_upper_limit,
            vtype=int,
            kwargs=kwargs,
        )

        assert (
            num_bits_lower_limit > 1
        ), "The lower bound for the bit resolution has to be > 1. 1 bit resolution is not supported and produces NaN."

        self.dataloader_generator = dataloader_generator

        self.num_bits_upper_limit = num_bits_upper_limit
        self.num_bits_lower_limit = num_bits_lower_limit

    def _evaluate(self, index, layer_bit_nums, out, *args, **kwargs):
        algorithm: NSGA2 = kwargs.get('algorithm')

        logger.debug("Evaluating individual #{} of {} in Generation {}".format(
            index + 1, algorithm.pop_size, algorithm.n_iter))
        bits_str = [str(x) for x in layer_bit_nums]
        logger.debug(f"\tBit widths: {bits_str}")

        self.model.bit_widths = layer_bit_nums

        f1_accuracy_objective = self.accuracy_function(
            self.model.base_model,
            self.dataloader_generator,
            progress=self.progress,
            title="Evaluating {}/{}".format(index + 1, algorithm.pop_size)
        )
        f2_quant_objective = self.model.get_bit_weighted()

        logger.debug(
            f"\t Evaluated, acc: {f1_accuracy_objective:.4f}, weighted bits: {f2_quant_objective}"
        )

        g1_accuracy_constraint = self.min_accuracy - f1_accuracy_objective

        # NOTE: In pymoo, each objective function is supposed to be minimized,
        # and each constraint needs to be provided in the form of <= 0
        out["F"] = [-f1_accuracy_objective, f2_quant_objective]
        out["G"] = [g1_accuracy_constraint]


prepare_exploration_function = prepare_quantization_problem
repair_method = RoundingRepair()
sampling_method = IntegerRandomSampling()
init_function = init_quant_model
update_params_function = update_quant_model_params
