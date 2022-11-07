"""
This module contains the problem definitions for the quantization exploration using pymoo.
"""
import numpy as np
from tqdm import tqdm
from src.utils.logger import logger

from pymoo.core.problem import ElementwiseProblem, ElementwiseEvaluationFunction, LoopedElementwiseEvaluation
from pymoo.algorithms.moo.nsga2 import NSGA2
from src.quantization.quantized_model import QuantizedModel


class ElementwiseEvaluationFunctionWithIndex(ElementwiseEvaluationFunction):
    def __init__(self, problem, args, kwargs) -> None:
        super().__init__(problem, args, kwargs)
        
    def __call__(self, i, x):
        out = dict()
        self.problem._evaluate(i, x, out, *self.args, **self.kwargs)
        return out
        

class LoopedElementwiseEvaluationWithIndex(LoopedElementwiseEvaluation):
    def __call__(self, f, X):
        algorithm:NSGA2 = f.kwargs.get('algorithm')
        pbar = tqdm(total=len(X), position=1, desc="Generation {}".format(algorithm.n_iter))
        results = []
        for i, x in enumerate(X):
            results.append(f(i, x))
            pbar.update(1)
        pbar.close()
        return results
        

class LayerwiseQuantizationProblem(ElementwiseProblem):
    """
    A pymoo problem defenition for the quantization exploration.
    """

    def __init__(
        self,
        qmodel:QuantizedModel,
        dataloader_generator,
        accuracy_func,
        progress=True,
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
            n_var=len(qmodel.quantizer_modules),
            n_constr=1,  # accuracy constraint
            n_obj=2,  # accuracy and low bit num
            xl=num_bits_lower_limit,
            xu=num_bits_upper_limit,
            vtype=int,
            elementwise_func=ElementwiseEvaluationFunctionWithIndex,
            elementwise_runner=LoopedElementwiseEvaluationWithIndex(),
            kwargs=kwargs,
        )

        assert (
            num_bits_lower_limit > 1
        ), "The lower bound for the bit resolution has to be > 1. 1 bit resolution is not supported and produces NaN."

        self.qmodel = qmodel
        self.dataloader_generator = dataloader_generator
        self.accuracy_func = accuracy_func
        
        self.progress = progress

        self.min_accuracy = min_accuracy
        self.num_bits_upper_limit = num_bits_upper_limit
        self.num_bits_lower_limit = num_bits_lower_limit

    def _evaluate(self, index, layer_bit_nums, out, *args, **kwargs):

        algorithm: NSGA2 = kwargs.get('algorithm')

        logger.info("Evaluating individual #{} of {} in Generation {}".format(
            index + 1, algorithm.pop_size, algorithm.n_iter
        ))

        self.qmodel.bit_widths = layer_bit_nums
        data_loader = self.dataloader_generator.get_dataloader()
        
        f1_accuracy_objective = self.accuracy_func(self.qmodel.model, data_loader, progress=self.progress, 
                                                   title="Evaluating {}/{}".format(index + 1, algorithm.pop_size))
        f2_quant_objective = self.qmodel.get_bit_weighted()

        logger.info(f"Evaluated individual, accuracy: {f1_accuracy_objective}")

        g1_accuracy_constraint = self.min_accuracy - f1_accuracy_objective

        # NOTE: In pymoo, each objective function is supposed to be minimized,
        # and each constraint needs to be provided in the form of <= 0
        out["F"] = [-f1_accuracy_objective, f2_quant_objective]
        out["G"] = [g1_accuracy_constraint]
