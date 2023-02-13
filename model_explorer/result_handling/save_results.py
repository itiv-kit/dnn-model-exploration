import os
import pickle
import pandas as pd

from datetime import datetime

from model_explorer.utils.logger import logger


RESULTS_DIR = "./results"


def save_result_pickle(res, problem_name, model_name, dataset_name):
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

    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        filename = 'expl_{}_{}_{}_{}_slurmid_{}.pkl'.format(
            problem_name, model_name, dataset_name, date_str, os.environ['SLURM_ARRAY_TASK_ID']
        )
    else:
        filename = 'expl_{}_{}_{}_{}.pkl'.format(
            problem_name, model_name, dataset_name, date_str
        )

    filename = os.path.join(RESULTS_DIR, filename)

    with open(filename, "wb") as res_file:
        pickle.dump(res, res_file)

    logger.info(f"Saved result object to: {filename}")


def save_results_df_to_csv(name: str, result_df: pd.DataFrame, 
                           problem_name: str, model_name: str,
                           dataset_name: str):
    # store results in csv
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        filename = '{}_{}_{}_{}_{}_slurmid_{}.csv'.format(
            name, problem_name, model_name, dataset_name, date_str, os.environ['SLURM_ARRAY_TASK_ID']
        )
    else:
        filename = '{}_{}_{}_{}_{}.csv'.format(
            name, problem_name, model_name, dataset_name, date_str
        )
    filename = os.path.join(RESULTS_DIR, filename)

    result_df.to_csv(filename)
    logger.info(f"Saved result csv to: {filename}")
