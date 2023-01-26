"""
Run a quantization exploration from a workload .yaml file
Usage:
    $ python path/to/main.py --workloads resnet50.yaml
Usage:
    $ python path/to/main.py --workloads fcn-resnet50.yaml      # TorchVision: fully connected etesNet50
                                             resnet50.yaml          # TorchVision: ResNet50
                                         yolov5.yaml            # TorchVision: YoloV5
                                         lenet5.yaml            # Custom: LeNet5
"""
import os
import argparse
import numpy as np
from datetime import datetime
import pickle

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from model_explorer.utils.logger import logger
from model_explorer.utils.setup import build_dataloader_generators, setup_torch_device, \
        setup_workload, get_prepare_exploration_function
from model_explorer.utils.workload import Workload


RESULTS_DIR = "./results"



def save_result(res, problem_name, model_name, dataset_name):
    """Save the result object from the exploration as a pickle file.

    Args:
        res (obj):
            The result object to save.
        model_name (str):
            The name of the model this result object belongs to.
            This is used as a prefix for the saved file.
    """
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")

    filename = 'expl_{}_{}_{}_{}.pkl'.format(
        problem_name, model_name, dataset_name, date_str)

    with open(os.path.join(RESULTS_DIR, filename), "wb") as res_file:
        pickle.dump(res, res_file)

    logger.info(f"Saved result object to: {filename}")



def explore_model(workload: Workload,
                  skip_baseline: bool,
                  progress: bool,
                  verbose: bool) -> None:
    dataloaders = build_dataloader_generators(workload['exploration']['datasets'])
    model, accuracy_function = setup_workload(workload['model'])
    device = setup_torch_device()

    if not skip_baseline:
        # collect model basline information
        baseline_dataloader = dataloaders['baseline']
        logger.info("Collecting baseline...")
        baseline = accuracy_function(model, baseline_dataloader, title="Baseline Generation")
        logger.info(f"Done. Baseline accuracy: {baseline:.3f}")

    prepare_function, repair_method, sampling_method = get_prepare_exploration_function(workload['problem']['problem_function'])
    kwargs: dict = workload['exploration']['extra_args']
    if 'calibration' in workload.yaml_data:
        kwargs['calibration_file'] = workload['calibration']['file']
    min_accuracy = workload['exploration']['minimum_accuracy']
    problem = prepare_function(model, device, dataloaders['exploration'],
                               accuracy_function, min_accuracy,
                               verbose, progress, **kwargs)

    crossover = SBX(prob_var=workload['exploration']['nsga']['crossover_prob'],
                    eta=workload['exploration']['nsga']['crossover_eta'],
                    repair=repair_method,
                    vtype=float)
    mutation = PolynomialMutation(prob=workload['exploration']['nsga']['mutation_prob'],
                                  eta=workload['exploration']['nsga']['mutation_eta'],
                                  repair=repair_method)

    print(sampling_method)

    algorithm = NSGA2(
        pop_size=workload['exploration']['nsga']['pop_size'],
        n_offsprings=workload['exploration']['nsga']['offsprings'],
        sampling=sampling_method,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True,
    )

    termination = get_termination("n_gen", workload['exploration']['nsga']['generations'])

    logger.info("Starting problem minimization.")

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=1,
        save_history=True,
        verbose=True
    )

    logger.info("Finished problem minimization.")

    if res.F is None:
        logger.warning("No solutions found for the given constraints.")
        return

    # since we inverted our objective functions we have to invert the result back
    res.F = np.abs(res.F)

    return res



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "workload",
        help="The path to the workload yaml file.")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show verbose information.")
    parser.add_argument(
        "-p",
        "--progress",
        action="store_true",
        help="Show the current inference progress.")
    parser.add_argument(
        "-s",
        "--skip-baseline",
        help="skip baseline computation to save time",
        action="store_true"
    )
    opt = parser.parse_args()

    logger.info("Model exploration started")

    workload_file = opt.workload
    if os.path.isfile(workload_file):
        workload = Workload(workload_file)
        results = explore_model(workload, opt.skip_baseline, opt.progress, opt.verbose)
        save_result(results, workload['problem']['problem_function'], workload['model']['type'], workload['exploration']['datasets']['exploration']['type'])

    else:
        logger.warning("Declared workload file could not be found.")
        raise Exception(f"No file {opt.workload} found.")

    logger.info("Model exploration finished")

