from pymoo.core.problem import ElementwiseProblem, looped_eval

from utils.sparsity_metrics import set_configuration, store_current_pass, \
    get_current_pass_sparse_created_mean, add_accuracy_and_timing
from utils.custom_timeit import timings
import logging

class SparsityThresholdProblem(ElementwiseProblem):
    """A pymoo problem defenition for the sparsity exploration.
    """

    def __init__(self,
                sparse_model_generator,
                data_loader_generator,
                func_eval=looped_eval,
                runner=None,
                discrete_threshold_steps=1,
                discrete_threshold_method=None,
                sample_limit = None,
                x_upper_limit = 0.4, #upper limit of threshold raw values
                threshold_limit = 1.0,
                min_accuracy = 0,
                base_network_name = None,
                base_dataset_name = None,
                **kwargs):
        """Inits a sparsety exploration problem.

        Args:
            sparse_model_generator (SparseModelGenerator):
                The generator to provide and update the sparse model that is to be evaluated.
            data_loader_generator (DataLoaderGenerator):
                The generator to provide each evaluation with a fresh data loader.
            func_eval (callable, optional):
                The function that calls func_elementwise_eval for
                ALL solutions to be evaluated. Defaults to looped_eval.
            runner (callable, optional):
                One of the two ways of parallelization which are supported py pymoo.
                Defaults to None.
            discrete_threshold_steps (int, optional):
                Number of steps the thresholds should be divided into.
                Defaults to 1.
            discrete_threshold_method (str, optional):
                The method used to generate the discret thresholds.
                Defaults to None.
            sample_limit (int, optional):
                The sample limit for all data loaders.
                Defaults to None.
            x_upper_limit (float, optional):
                The upper limit for the thresholds.
                Defaults to 0.4.
            min_accuracy (int, optional):
                The minimum accuracy the model should reach.
                This is set as a problem constraint.
                Defaults to 0.
            base_network_name (str, optional):
                The name of the base network used.
                Defaults to None.
            base_dataset_name (str, optional):
                The name of the dataset used.
                Defaults to None.
        """
        super().__init__(n_var=sparse_model_generator.get_layer_count(),
                         n_constr=1,
                         n_obj=2,
                         xl=0,
                         xu=x_upper_limit,
                         func_eval=func_eval,
                         runner=runner,
                         kwargs=kwargs)

        self.sparse_model_generator = sparse_model_generator
        self.data_loader_generator = data_loader_generator

        self.sample_limit = sample_limit
        self.config_counter = 0
        self.min_accuracy = min_accuracy
        self.progress = kwargs.get('progress', False)

        self.discrete_threshold_steps = discrete_threshold_steps
        self.discrete_threshold_method = discrete_threshold_method
        self.threshold_limit = threshold_limit

        # save for image generation
        self.base_network_name = base_network_name
        self.base_dataset_name = base_dataset_name


    def _evaluate(self, thresholds, out, *args, **kwargs):

        set_configuration(self.config_counter) # set the id/ number of this configuration

        if self.discrete_threshold_method is not None:
            if self.discrete_threshold_method == 'linear':
                discrete_thresholds = [(t / self.discrete_threshold_steps) *
                    self.threshold_limit for t in thresholds]
            elif self.discrete_threshold_method == 'log':
                pass
        else:
            discrete_thresholds = thresholds

        sparse_model = self.sparse_model_generator.update_sparse_model(discrete_thresholds)
        data_loader = self.data_loader_generator.get_data_loader(self.sample_limit)

        f1_accuracy_objective = sparse_model.compute_accuracy(data_loader, progress=self.progress).item()

        # f2 is the mean of created sparse blocks
        f2_sparsity_objective = get_current_pass_sparse_created_mean()

        self.config_counter += 1

        g1_accuracy_constraint = self.min_accuracy - f1_accuracy_objective

        logging.getLogger('sparsity-analysis').info(f"Current Accuracy: {f1_accuracy_objective}")
        logging.getLogger('sparsity-analysis').info(f"Sparse Blocks: {f2_sparsity_objective}")

        # FIXME: make more flexible
        acc_funcs = ['compute_detection_accuracy', 'compute_segmentation_accuracy', 'compute_classification_accuracy']

        time_acc = None

        for func in acc_funcs:
            if func in timings:
                time_acc = timings[func]
                break

        add_accuracy_and_timing(len(data_loader.dataset), f1_accuracy_objective, time_acc)

        store_current_pass()

        # NOTE: In pymoo, each objective function is supposed to be minimized,
        # and each constraint needs to be provided in the form of <= 0
        # this is why we make our objectives negative
        out["F"] = [-f1_accuracy_objective, -f2_sparsity_objective]
        out["G"] = [g1_accuracy_constraint]
