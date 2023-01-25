import torch
import importlib
import sys

from torch import nn
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling

from model_explorer.models.sparse_model import SparseModel
from model_explorer.utils.logger import logger
from model_explorer.problems.custom_problem import CustomExplorationProblem
from model_explorer.utils.data_loader_generator import DataLoaderGenerator


def prepare_sparsity_problem(model: nn.Module, device: torch.device,
                             dataloader_generator: DataLoaderGenerator,
                             accuracy_function: callable, min_accuracy: float,
                             verbose: bool, progress: bool,
                             **kwargs: dict):
    block_size = kwargs.get('block_size')
    discrete_threshold_steps = kwargs.get('discrete_threshold_steps')
    discrete_threshold_method = kwargs.get('discrete_threshold_method')
    threshold_limit = kwargs.get('threshold_limit')

    smodel = SparseModel(model, block_size, device, verbose)
    logger.debug("Initalized sparse model with {} sparse modules".format(smodel.get_explorable_parameter_count()))
    logger.info("Sparsity problem and model initialized")

    return SparsityThresholdProblem(
        sparse_model=smodel,
        dataloader_generator=dataloader_generator,
        accuracy_function=accuracy_function,
        min_accuracy=min_accuracy,
        progress=progress,
        discrete_threshold_steps=discrete_threshold_steps,
        discrete_threshold_method=discrete_threshold_method,
        threshold_limit=threshold_limit
    )


class SparsityThresholdProblem(CustomExplorationProblem):
    """A pymoo problem defenition for the sparsity exploration.
    """

    def __init__(
            self,
            sparse_model: SparseModel,
            dataloader_generator: DataLoaderGenerator,
            accuracy_function: callable,
            min_accuracy: float,
            progress: bool,
            discrete_threshold_steps: int,
            discrete_threshold_method: str,
            threshold_limit: float,
            **kwargs):
        """Inits a sparsity exploration problem.
        """
        super().__init__(
            model=sparse_model,
            accuracy_function=accuracy_function,
            progress=progress,
            min_accuracy=min_accuracy,
            n_var=sparse_model.get_explorable_parameter_count(),
            n_constr=1,
            n_obj=2,
            xl=0,
            xu=threshold_limit,
            vtype=float,
            kwargs=kwargs
        )

        self.dataloader_generator = dataloader_generator

        self.discrete_threshold_steps = discrete_threshold_steps
        self.discrete_threshold_method = discrete_threshold_method
        self.threshold_limit = threshold_limit


    def _evaluate(self, index, thresholds, out, *args, **kwargs):
        algorithm: NSGA2 = kwargs.get('algorithm')

        # if self.discrete_threshold_method is not None:
        #     if self.discrete_threshold_method == 'linear':
        #         discrete_thresholds = [
        #             (t / self.discrete_threshold_steps) * self.threshold_limit
        #             for t in thresholds
        #         ]
        #     elif self.discrete_threshold_method == 'log':
        #         pass
        # else:
        #     discrete_thresholds = thresholds

        logger.debug("Evaluating individual #{} of {} in Generation {}".format(
            index + 1, algorithm.pop_size, algorithm.n_iter))
        threshold_strs = ['{:.2f}'.format(x) for x in thresholds]
        logger.debug(f"\tThesholds: {threshold_strs}")

        self.model.thresholds = thresholds

        f1_accuracy_objective = self.accuracy_function(
            self.model.base_model,
            self.dataloader_generator,
            progress=self.progress,
            title="Evaluating {}/{}".format(index + 1, algorithm.pop_size)
        )
        # f2 is the mean of created sparse blocks
        f2_sparsity_objective = self.model.get_total_created_sparse_blocks()

        g1_accuracy_constraint = self.min_accuracy - f1_accuracy_objective

        logger.debug(
            f"\tEvaluated, acc: {f1_accuracy_objective:.4f}, " +
            f"sparse blks created: {f2_sparsity_objective}"
        )

        # NOTE: In pymoo, each objective function is supposed to be minimized,
        # and each constraint needs to be provided in the form of <= 0
        out["F"] = [-f1_accuracy_objective, -f2_sparsity_objective]
        out["G"] = [g1_accuracy_constraint]

        self.model.reset_model_stats()


prepare_exploration_function = prepare_sparsity_problem
repair_method = None
sampling_method = FloatRandomSampling()
