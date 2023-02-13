import os
import argparse
import collections.abc

from model_explorer.utils.logger import logger
from model_explorer.utils.workload import Workload
from model_explorer.exploration.explore_model import explore_model
from model_explorer.result_handling.save_results import save_result_pickle

# maps SLURM_JOB_ID to the accoring workload settings
extra_slurm_args = {
    1: {'exploration': {'nsga': {'mutation_eta': 10}}}
}


# https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
def deep_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


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

        # Update workload with extra args if running with different nodes on a slurm cluster
        if 'SLURM_ARRAY_TASK_ID' in os.environ:
            job_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
            if job_id in extra_slurm_args:
                workload.update(extra_slurm_args[job_id])

        results = explore_model(workload, opt.skip_baseline, opt.progress, opt.verbose)

        save_result_pickle(results, workload['problem']['problem_function'],
                           workload['model']['type'], workload['exploration']['datasets']['exploration']['type'])

    else:
        logger.warning("Declared workload file could not be found.")
        raise Exception(f"No file {opt.workload} found.")

    logger.info("Model exploration finished")

