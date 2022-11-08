"""
Run a quantization exploration from a workload .yaml file
Usage:
    $ python path/to/main.py --workloads resnet50.yaml
Usage:
    $ python path/to/main.py --workloads fcn-resnet50.yaml      # TorchVision: fully connected ResNet50
                                         resnet50.yaml          # TorchVision: ResNet50
                                         yolov5.yaml            # TorchVision: YoloV5
                                         lenet5.yaml            # Custom: LeNet5
"""
import os
import argparse
import numpy as np
from datetime import datetime
import pickle

# import troch quantization and activate the replacement of modules
from pytorch_quantization import quant_modules
quant_modules.initialize()

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.evaluator import Evaluator

from src.utils.logger import logger
from src.utils.setup import setup
from src.utils.workload import Workload
from src.exploration.weighting_functions import bits_weighted_linear
from src.utils.data_loader_generator import DataLoaderGenerator

from src.exploration.problems import LayerwiseQuantizationProblem
from src.quantization.quantized_model import QuantizedModel


RESULTS_DIR = "./results"


def save_result(res, model_name, dataset_name):
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

    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    filename = 'exploration_{}_{}_{}.pkl'.format(
        model_name, dataset_name, date_str)

    with open(os.path.join(RESULTS_DIR, filename), "wb") as res_file:
        pickle.dump(res, res_file)

    logger.info(f"Saved result object to: {filename}")



def explore_quantization(workload: Workload, calibration_file: str, progress: bool, verbose: bool) -> None:
    """Runs the given workload.

    Args:
        workload (Workload):
            The workload loaded from a workload yaml file.
        collect_baseline (bool):
            Whether to collect basline metrics of the model without quantization.
    """
    model, accuracy_function, dataset, collate_fn, device = setup(workload)
    dataloader_generator = DataLoaderGenerator(dataset, 
                                               collate_fn, 
                                               batch_size=workload['exploration']['batch_size'],
                                               limit=workload['exploration']['sample_limit'])

    # now switch to quantized model
    qmodel = QuantizedModel(model, device, 
                            weighting_function=bits_weighted_linear,
                            verbose=verbose)
    logger.info("Added {} Quantizer modules to the model".format(len(qmodel.quantizer_modules)))

    # collect model basline information
    baseline_data_loader = dataloader_generator.get_dataloader()
    logger.info("Collecting baseline...")
    qmodel.disable_quantization()
    baseline = accuracy_function(qmodel.model, baseline_data_loader, len(dataloader_generator), title="Baseline Generation")
    qmodel.enable_quantization()
    logger.info(f"Done. Baseline accuracy: {baseline}")

    # Load the previously generated calibration file
    logger.info(f"Loading calibration file: {calibration_file}")
    qmodel.load_calibration(calibration_file)

    # configure exploration
    # FIXME: add to workload file
    problem = LayerwiseQuantizationProblem(
        qmodel,
        dataloader_generator,
        accuracy_function,
        num_bits_upper_limit=16,
        num_bits_lower_limit=3,
        min_accuracy=0.70,
        progress=progress
    )

    # TODO put into own module and pass args from workload
    # TODO set through workload
    sampling = IntegerRandomSampling()
    crossover = SBX(prob_var=workload['exploration']['nsga']['crossover_prob'],
                    eta=workload['exploration']['nsga']['crossover_eta'],
                    repair=RoundingRepair(), 
                    vtype=float)
    mutation = PolynomialMutation(prob=workload['exploration']['nsga']['mutation_prob'],
                                  eta=workload['exploration']['nsga']['mutation_eta'],
                                  repair=RoundingRepair())

    algorithm = NSGA2(
        pop_size=workload['exploration']['nsga']['pop_size'],
        n_offsprings=workload['exploration']['nsga']['offsprings'],
        sampling=sampling,
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

    parser.add_argument("calibration_file")
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
        "-fn",
        "--filename",
        help="override default filename for calibration pickle file")
    opt = parser.parse_args()

    logger.info("Quantization Exploration Started")

    workload_file = opt.workload
    if os.path.isfile(workload_file):
        workload = Workload(workload_file)
        results = explore_quantization(workload, opt.calibration_file, opt.progress, opt.verbose)
        save_result(results, workload['model']['type'], workload['dataset']['type'])

    else:
        logger.warning("Declared workload file could not be found.")
        raise Exception(f"No file {opt.workload} found.")

    logger.info("Quantization Exploration Finished")

