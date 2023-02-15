import torch
import numpy as np

from torch import nn
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling

from model_explorer.models.sparse_model import SparseModel
from model_explorer.utils.logger import logger
from model_explorer.problems.custom_problem import CustomExplorationProblem
from model_explorer.utils.data_loader_generator import DataLoaderGenerator


def update_sparse_model_params(model: SparseModel, thresholds: list):
    model.thresholds = thresholds


def init_sparse_model(model: nn.Module, device: torch.device,
                      **kwargs: dict) -> SparseModel:
    """Create sparse model

    Args:
        model (nn.Module): base model, which is replaced with Sparse Modules
        device (torch.device): torch device
    """
    block_size = kwargs.get('block_size')
    sparse_model = SparseModel(model, block_size, device)
    logger.debug("Initialized sparse model with {} sparse modules".format(sparse_model.get_explorable_parameter_count()))
    return sparse_model


def prepare_sparsity_problem(model: nn.Module, device: torch.device,
                             dataloader_generator: DataLoaderGenerator,
                             accuracy_function: callable, min_accuracy: float,
                             progress: bool, **kwargs: dict):
    """Generate sparsity exploration problem for NSGA2

    Args:
        model (nn.Module): base model to explore
        device (torch.device): torch device
        dataloader_generator (DataLoaderGenerator): Dataset used for exploration
        accuracy_function (callable): accuracy function that gets evaluated
        min_accuracy (float): minimum accuracy constraint
        progress (bool): show progress?
    """
    discrete_threshold_steps = kwargs.get('discrete_threshold_steps')
    discrete_threshold_method = kwargs.get('discrete_threshold_method')
    threshold_limit = kwargs.get('threshold_limit')

    sparse_model = init_sparse_model(model, device, **kwargs)

    logger.info("Sparsity problem and model initialized")

    return SparsityThresholdProblem(
        sparse_model=sparse_model,
        dataloader_generator=dataloader_generator,
        accuracy_function=accuracy_function,
        min_accuracy=min_accuracy,
        progress=progress,
        discrete_threshold_steps=discrete_threshold_steps,
        discrete_threshold_method=discrete_threshold_method,
        threshold_limit=threshold_limit
    )


class SparsityThresholdProblem(CustomExplorationProblem):
    """A pymoo problem definition for the sparsity exploration.
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

        logger.debug("Evaluating individual #{} of {} in Generation {}".format(
            index + 1, algorithm.pop_size, algorithm.n_iter))
        threshold_strs = ['{:.3f}'.format(x) for x in thresholds]
        logger.debug(f"\tThresholds: {threshold_strs}")

        self.model.thresholds = thresholds

        f1_accuracy_objective = self.accuracy_function(
            self.model.base_model,
            self.dataloader_generator,
            progress=self.progress,
            title="Evaluating {}/{}".format(index + 1, algorithm.pop_size)
        )
        # get total created returns all created for a given batch, therefore div by batch size
        f2_sparsity_objective = self.model.get_total_created_sparse_blocks()
        f2_sparsity_objective /= len(self.dataloader_generator)

        g1_accuracy_constraint = self.min_accuracy - f1_accuracy_objective

        logger.debug(
            f"\tEvaluated, acc: {f1_accuracy_objective:.4f}, " +
            f"sparse blocks created: {f2_sparsity_objective}"
        )

        # NOTE: In pymoo, each objective function is supposed to be minimized,
        # and each constraint needs to be provided in the form of <= 0
        out["F"] = [-f1_accuracy_objective, -f2_sparsity_objective]
        out["G"] = [g1_accuracy_constraint]

        self.model.reset_model_stats()


class FloatRandomSamplingWithDefinedIndividual(FloatRandomSampling):
    """A modified version of FloatRandomSampling, which is able to add predefined individuals to the population
    """

    def __init__(self, var_type=np.float64, predefined: list = []) -> None:
        super().__init__()
        np.random.seed(None)
        self.var_type = var_type
        self.predefined = predefined

    def _do(self, problem, n_samples, **kwargs):
        pop = super()._do(problem, n_samples, **kwargs)
        pop = pop[len(self.predefined):]

        for predef in self.predefined:
            predef_ind = np.full((1, problem.n_var), predef)
            pop = np.concatenate((predef_ind, pop))
        return pop


prepare_exploration_function = prepare_sparsity_problem
repair_method = None
sampling_method = FloatRandomSamplingWithDefinedIndividual(predefined=[0.15])
init_function = init_sparse_model
update_params_function = update_sparse_model_params
