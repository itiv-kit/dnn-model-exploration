import os
import argparse
from datetime import datetime

from model_explorer.utils.logger import logger
from model_explorer.utils.workload import Workload
from model_explorer.exploration.explore_model import explore_model
from model_explorer.result_handling.save_results import save_result_pickle




def sweep_ga_parameters(workload):
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

            result = explore_model(workload, skip_baseline, progress=False, verbose=False)

            filename = os.path.join(result_dir, 'result_muteta_{}_croeta_{}.pkl'.format(mutation_eta, crossover_eta))
            save_result_pickle(result, overwrite_filename=filename)

            logger.info("Done saved at {}".format(filename))
            logger.info("#"*80)

            skip_baseline = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "workload",
        help="The path to the workload yaml file.")
    opt = parser.parse_args()

    logger.info("Quantization Exploration Started")

    workload_file = opt.workload
    if os.path.isfile(workload_file):
        workload = Workload(workload_file)
        sweep_ga_parameters(workload)

    else:
        logger.warning("Declared workload file could not be found.")
        raise Exception(f"No file {opt.workload} found.")

    logger.info("Quantization GA Sweep Exploration Finished")
