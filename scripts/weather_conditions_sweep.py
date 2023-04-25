import os
import argparse
import logging

from datetime import datetime

from model_explorer.utils.logger import logger, set_console_logger_level
from model_explorer.utils.workload import Workload
from model_explorer.exploration.explore_model import explore_model
from model_explorer.result_handling.save_results import save_result_pickle





def sweep_ga_parameters(workload):
    """Perform a sweep over a set of given mutation and crossover ETAs to get an
    idea of how they change the network convergence.

    Args:
        workload (_type_): _description_
    """
    skip_baseline = True

    result_dir = 'results/exploration_sweep_{}'.format(datetime.now().strftime("%Y-%m-%d_%H-%M"))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for c_index, combination in enumerate([[0.03, 0.015, 0.002], [0.02, 0.01, 0.005], [0.01, 0.005, 0.01]]):
        for pattern in list(range(1, 13, 1)):
            workload['exploration']['datasets']['exploration']['alpha'] = combination[0]
            workload['exploration']['datasets']['exploration']['beta'] = combination[1]
            workload['exploration']['datasets']['exploration']['dropsize'] = combination[2]
            workload['exploration']['datasets']['exploration']['pattern'] = pattern

            logger.info("#"*80)
            logger.info("Running Sweep point with combi={}, pattern={}".format(combination, pattern))
            logger.info("#"*80)

            result = explore_model(workload, skip_baseline, progress=False)

            filename = os.path.join(result_dir, 'result_pattern_{}_comb_{}.pkl'.format(pattern, c_index))
            save_result_pickle(result, overwrite_filename=filename)

            logger.info("Done saved at {}".format(filename))
            logger.info("#"*80)

            skip_baseline = True


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
    opt = parser.parse_args()

    if opt.verbose:
        set_console_logger_level(level=logging.DEBUG)

    logger.info("Exploration sweep over GA parameters started")

    workload_file = opt.workload
    if not os.path.isfile(workload_file):
        logger.warning("Declared workload file could not be found.")
        raise Exception(f"No file {opt.workload} found.")

    workload = Workload(workload_file)
    sweep_ga_parameters(workload)

    logger.info("GA sweep exploration finished")