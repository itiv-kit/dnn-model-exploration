import os
import pickle
import pandas as pd
import pymoo.core.result

from datetime import datetime

from model_explorer.utils.logger import logger


RESULTS_DIR = "./results"


def save_result_pickle(res: pymoo.core.result.Result,
                       problem_name: str = "",
                       model_name: str = "",
                       dataset_name: str = "",
                       overwrite_filename: str = ""):
    """Saves a pymoo result directly into a pickle. Files are stored in the
    RESULTS_DIR (usually ./results). Filename also includes the slurm job id if
    present.

    Args:
        res (pymoo.core.result.Result): Input Result
        problem_name (str, optional): Name of the problem, will be appended to the filename. Defaults to "".
        model_name (str, optional): Model name, will be appended to the filename. Defaults to "".
        dataset_name (str, optional): Dataset name, will be appended to the filename. Defaults to "".
        overwrite_filename (str, optional): Option to give a custom filename. Defaults to "".
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
    if overwrite_filename != "":
        filename = overwrite_filename

    with open(filename, "wb") as res_file:
        pickle.dump(res, res_file)

    logger.info(f"Saved result object to: {filename}")


def save_results_df_to_csv(name: str, result_df: pd.DataFrame,
                           problem_name: str, model_name: str,
                           dataset_name: str):
    """Store Dataframes as CSVs, it also adds the slurm job id if present

    Args:
        name (str): Name of file
        result_df (pd.DataFrame): Input dataframe
        problem_name (str): Name of the problem, will be appended to the filename
        model_name (str): Name of the model, will be appended to the filename
        dataset_name (str): Name of the dataset, will be appended to the filename
    """

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
