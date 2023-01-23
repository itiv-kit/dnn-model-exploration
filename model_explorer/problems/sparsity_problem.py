import torch
import importlib

from pymoo.algorithms.moo.nsga2 import NSGA2

from model_explorer.models.sparse_model import SparseModel
from model_explorer.utils.logger import logger
from model_explorer.problems.custom_problem import CustomExplorationProblem
from model_explorer.utils.data_loader_generator import DataLoaderGenerator


class SparsityThresholdProblem(CustomExplorationProblem):
    """A pymoo problem defenition for the sparsity exploration.
    """

    def __init__(
            self,
            sparse_model: SparseModel,
            dataloader_generator: DataLoaderGenerator,
            discrete_threshold_steps=1,
            discrete_threshold_method=None,
            sample_limit=None,
            x_upper_limit=0.4,  # upper limit of threshold raw values
            threshold_limit=1.0,
            min_accuracy=0,
            **kwargs):
        """Inits a sparsity exploration problem.
        """
        super().__init__(n_var=sparse_model.get_layer_count(),
                         n_constr=1,
                         n_obj=2,
                         xl=0,
                         xu=x_upper_limit,
                         kwargs=kwargs)

        self.sparse_model = sparse_model
        self.dataloader_generator = dataloader_generator

        self.sample_limit = sample_limit
        self.config_counter = 0
        self.min_accuracy = min_accuracy
        self.progress = kwargs.get('progress', False)

        self.discrete_threshold_steps = discrete_threshold_steps
        self.discrete_threshold_method = discrete_threshold_method
        self.threshold_limit = threshold_limit


    def _evaluate(self, thresholds, out, *args, **kwargs):
        algorithm: NSGA2 = kwargs.get('algorithm')

        if self.discrete_threshold_method is not None:
            if self.discrete_threshold_method == 'linear':
                discrete_thresholds = [
                    (t / self.discrete_threshold_steps) * self.threshold_limit
                    for t in thresholds
                ]
            elif self.discrete_threshold_method == 'log':
                pass
        else:
            discrete_thresholds = thresholds

        sparse_model = self.sparse_model_generator.update_sparse_model(
            discrete_thresholds)
        data_loader = self.data_loader_generator.get_data_loader(
            self.sample_limit)

        f1_accuracy_objective = sparse_model.compute_accuracy(
            data_loader, progress=self.progress).item()

        # f2 is the mean of created sparse blocks
        f2_sparsity_objective = get_current_pass_sparse_created_mean()

        self.config_counter += 1

        g1_accuracy_constraint = self.min_accuracy - f1_accuracy_objective

        logging.getLogger('sparsity-analysis').info(
            f"Current Accuracy: {f1_accuracy_objective}")
        logging.getLogger('sparsity-analysis').info(
            f"Sparse Blocks: {f2_sparsity_objective}")

        # FIXME: make more flexible
        acc_funcs = [
            'compute_detection_accuracy', 'compute_segmentation_accuracy',
            'compute_classification_accuracy'
        ]

        add_accuracy_and_timing(len(data_loader.dataset),
                                f1_accuracy_objective, time_acc)

        store_current_pass()

        # NOTE: In pymoo, each objective function is supposed to be minimized,
        # and each constraint needs to be provided in the form of <= 0
        # this is why we make our objectives negative
        out["F"] = [-f1_accuracy_objective, -f2_sparsity_objective]
        out["G"] = [g1_accuracy_constraint]
