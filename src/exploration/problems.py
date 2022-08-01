"""
This module contains the problem definitions for the quantization exploration using pymoo.
"""
import numpy as np
from src.utils.logger import logger

from pymoo.core.problem import ElementwiseProblem
from src.quantization.quantization import QuantizedActivationsModel


class LayerwiseQuantizationProblem(ElementwiseProblem):
    """
    A pymoo problem defenition for the quantization exploration.
    """

    def __init__(
        self,
        quantization_model: QuantizedActivationsModel,
        data_loader_generator,
        accuracy_func,
        sample_limit=None,
        num_bits_upper_limit=8,
        num_bits_lower_limit=2,
        min_accuracy=0.3,
        **kwargs,
    ):
        """Inits a quantization exploration problem.

        Args:
            data_loader_generator (DataLoaderGenerator):
                The generator to provide each evaluation with a fresh data loader.
            sample_limit (int, optional):
                The sample limit for all data loaders.
                Defaults to None.
            min_accuracy (int, optional):
                The minimum accuracy the model should reach.
                This is set as a problem constraint.
                Defaults to 0.
        """
        super().__init__(
            n_var=quantization_model.get_n_quantizers(),
            n_constr=1,  # accuracy constraint
            n_obj=2,  # accuracy and low bit num
            xl=num_bits_lower_limit,
            xu=num_bits_upper_limit,
            kwargs=kwargs,
            type_var=int,
        )

        assert (
            num_bits_lower_limit > 1
        ), "The lower bound for the bit resolution has to be > 1. 1 bit resolution is not supported and produces NaN."

        self.quantization_model = quantization_model
        self.data_loader_generator = data_loader_generator
        self.accuracy_func = accuracy_func

        self.sample_limit = sample_limit
        self.min_accuracy = min_accuracy

        self.num_bits_upper_limit = num_bits_upper_limit
        self.num_bits_lower_limit = num_bits_lower_limit

    def _evaluate(self, layer_bit_nums, out, *args, **kwargs):

        logger.info("Trying new layer bit resolutions.")

        self.quantization_model.update_layer_n_bits(layer_bit_nums)
        data_loader = self.data_loader_generator.get_data_loader(
            limit=self.sample_limit
        )

        f1_accuracy_objective = self.accuracy_func(self.quantization_model, data_loader)
        f2_quant_objective = np.sum(layer_bit_nums)

        logger.info(f"Achieved accuracy: {f1_accuracy_objective}")

        g1_accuracy_constraint = self.min_accuracy - f1_accuracy_objective

        # NOTE: In pymoo, each objective function is supposed to be minimized,
        # and each constraint needs to be provided in the form of <= 0
        out["F"] = [-f1_accuracy_objective, f2_quant_objective]
        out["G"] = [g1_accuracy_constraint]
