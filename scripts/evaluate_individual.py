import os
import argparse
import pandas as pd
import logging

from model_explorer.utils.logger import logger, set_console_logger_level
from model_explorer.utils.workload import Workload
from model_explorer.result_handling.collect_results import collect_results
from model_explorer.result_handling.save_results import save_results_df_to_csv
from model_explorer.exploration.evaluate_full_model import evaluate_full_model



def select_individuals(results_path: str, count: int) -> pd.DataFrame:
    results_collection = collect_results(results_path)

    results_collection.drop_duplicate_parameters()
    logger.debug("Loaded in total {} individuals".format(
        len(results_collection.individuals)))

    # select individuals based on a prodcut of normed F_0 and accuracy
    ind_df = results_collection.to_dataframe()
    ind_df['F_0'] = -ind_df['F_0']
    ind_df['norm_f0'] = ind_df['F_0'] / ind_df['F_0'].max()
    ind_df['norm_acc'] = ind_df['accuracy'] / ind_df['accuracy'].max()
    ind_df['weighted'] = ind_df['norm_f0'] * ind_df['norm_acc']

    ind_filtered = ind_df.sort_values(by=['weighted'], ascending=False)
    ind_filtered = ind_filtered.head(count)

    return ind_filtered


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("workload", help="The path to the workload yaml file.")
    parser.add_argument(
        "results_path",
        help="Path to the results file or folder to be evaluated")
    parser.add_argument('-n',
                        "--top_elements",
                        help="Select n individuals with the lowest bits",
                        type=int)
    parser.add_argument("-v",
                        "--verbose",
                        action="store_true",
                        help="Show verbose information.")
    parser.add_argument("-p",
                        "--progress",
                        action="store_true",
                        help="Show the current inference progress.")
    opt = parser.parse_args()

    logger.info("Reevaluation of individuals started")

    if opt.verbose:
        set_console_logger_level(level=logging.DEBUG)

    workload_file = opt.workload
    if not os.path.isfile(workload_file):
        logger.warning("Declared workload file could not be found.")
        raise Exception(f"No file {opt.workload} found.")

    workload = Workload(workload_file)
    individuals = select_individuals(opt.results_path, opt.top_elements)
    results = evaluate_full_model(workload, individuals, opt.progress)

    save_results_df_to_csv(
        'reeval', results,
        workload['problem']['problem_function'], workload['model']['type'],
        workload['reevaluation']['datasets']['reevaluate']['type']
    )

    logger.info("Reevaluation of individuals finished")
