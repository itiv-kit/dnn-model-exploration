import torch
import importlib

from pymoo.algorithms.moo.nsga2 import NSGA2
from torch import nn

from src.problems.custom_problem import CustomExplorationProblem
from src.utils.logger import logger
from src.models.quantized_model import QuantizedModel
from src.utils.data_loader_generator import DataLoaderGenerator


def prepare_quantization_problem(model: nn.module, device: torch.device,
                                 dataloader_generator: DataLoaderGenerator,
                                 verbose: bool, **kwargs: dict):
    num_bits_upper_limit = kwargs.get('num_bits_upper_limit')
    num_bits_lower_limit = kwargs.get('num_bits_lower_limit')
    weighting_function_name = kwargs.get('bit_weighting_function')
    calibration_file = kwargs.get('calibration_file')

    weighting_function = getattr(importlib.import_module('src.exploration.weighting_functions'),
                                 weighting_function_name, None)
    assert weighting_function is not None and callable(weighting_function), "error loading weighting function"

    qmodel = QuantizedModel(model, device,
                            weighting_function=weighting_function,
                            verbose=verbose)
    logger.debug("Added {} Quantizer modules to the model".format(len(qmodel.quantizer_modules)))

    # Load the previously generated calibration file
    logger.debug(f"Loading calibration file: {calibration_file}")
    qmodel.load_parameters(calibration_file)

    logger.info("Model loaded and initialized")

    return LayerwiseQuantizationProblem(qmodel=qmodel,
                                        dataloader_generator=dataloader_generator,
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
        num_bits_upper_limit: int = 8,
        num_bits_lower_limit: int = 2,
        **kwargs,
    ):
        """Inits a quantization exploration problem.
        """
        super().__init__(
            n_var=len(qmodel.quantizer_modules),
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

        self.qmodel = qmodel
        self.dataloader_generator = dataloader_generator
        
        self.num_bits_upper_limit = num_bits_upper_limit
        self.num_bits_lower_limit = num_bits_lower_limit

    def _evaluate(self, index, layer_bit_nums, out, *args, **kwargs):
        algorithm: NSGA2 = kwargs.get('algorithm')

        logger.debug("Evaluating individual #{} of {} in Generation {}".format(
            index + 1, algorithm.pop_size, algorithm.n_iter
        ))

        self.qmodel.bit_widths = layer_bit_nums

        f1_accuracy_objective = self.accuracy_func(self.qmodel.model, self.dataloader_generator, progress=self.progress,
                                                   title="Evaluating {}/{}".format(index + 1, algorithm.pop_size))
        f2_quant_objective = self.qmodel.get_bit_weighted()

        logger.debug(f"Evaluated individual, accuracy: {f1_accuracy_objective:.4f}, weighted bits: {f2_quant_objective}")

        g1_accuracy_constraint = self.min_accuracy - f1_accuracy_objective

        # NOTE: In pymoo, each objective function is supposed to be minimized,
        # and each constraint needs to be provided in the form of <= 0
        out["F"] = [-f1_accuracy_objective, f2_quant_objective]
        out["G"] = [g1_accuracy_constraint]


prepare_exploration_problem = prepare_quantization_problem
