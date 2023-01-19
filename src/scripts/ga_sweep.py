import os
import argparse
from datetime import datetime
import pickle

# import troch quantization and activate the replacement of modules
from pytorch_quantization import quant_modules
quant_modules.initialize()

from src.utils.logger import logger
from src.utils.workload import Workload
from src.scripts.explore import explore_quantization



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


def sweep_ga_parameters(workload, calibration_file):
    skip_baseline = False

    result_dir = 'results/exploration_sweep_{}'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for mutation_eta in [10, 30, 50, 100]:
        for crossover_eta in [15, 30, 45]:
            workload['exploration']['nsga']['mutation_eta'] = mutation_eta
            workload['exploration']['nsga']['crossover_eta'] = crossover_eta

            logger.info("#"*80)
            logger.info("Running Sweep point with mut_eta={}, crossover_eta={}".format(mutation_eta, crossover_eta))
            logger.info("#"*80)

            result = explore_quantization(workload, calibration_file,
                                          skip_baseline, progress=False, verbose=False)

            filename = os.path.join(result_dir, 'result_muteta_{}_croeta_{}.pkl'.format(mutation_eta, crossover_eta))
            with open(filename, 'wb') as res_file:
                pickle.dump(result, res_file)

            logger.info("Done saved at {}".format(filename))
            logger.info("#"*80)

            skip_baseline = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("calibration_file")
    parser.add_argument(
        "workload",
        help="The path to the workload yaml file.")
    opt = parser.parse_args()

    logger.info("Quantization Exploration Started")

    workload_file = opt.workload
    if os.path.isfile(workload_file):
        workload = Workload(workload_file)
        sweep_ga_parameters(workload, opt.calibration_file)

    else:
        logger.warning("Declared workload file could not be found.")
        raise Exception(f"No file {opt.workload} found.")

    logger.info("Quantization GA Sweep Exploration Finished")
