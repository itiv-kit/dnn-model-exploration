import os
import logging
import argparse

from model_explorer.utils.logger import logger, set_console_logger_level
from model_explorer.utils.workload import Workload
from model_explorer.exploration.explore_model import explore_model
from model_explorer.result_handling.save_results import save_result_pickle

"""Script to explore a model and to identify a well-fitting bit-width or sparsity threshold configuration.
"""

# maps SLURM_JOB_ID to the according workload settings
extra_slurm_args = {
    1: {'exploration': {'nsga': {'mutation_eta': 10}}}
}


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

    if opt.verbose:
        set_console_logger_level(level=logging.DEBUG)

    logger.info("Model exploration started")

    workload_file = opt.workload
    if not os.path.isfile(workload_file):
        logger.warning("Declared workload file could not be found.")
        raise Exception(f"No file {opt.workload} found.")

    workload = Workload(workload_file)

    # Update workload with extra args if running with different nodes on a slurm cluster
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        job_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
        if job_id in extra_slurm_args:
            workload.merge(extra_slurm_args[job_id])

    results = explore_model(workload, opt.skip_baseline, opt.progress, accuracy_constraint_baseline=True, accuracy_percentage_drop_allowance=0.03)

    save_result_pickle(results, workload['problem']['problem_function'],
                       workload['model']['type'], workload['exploration']['datasets']['exploration']['type'])

    logger.info("Model exploration finished")

