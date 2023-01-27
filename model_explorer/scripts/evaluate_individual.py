import os
import argparse
import pandas as pd
import importlib
from datetime import datetime

# import troch quantization and activate the replacement of modules
from pytorch_quantization import quant_modules

quant_modules.initialize()

from model_explorer.utils.logger import logger
from model_explorer.utils.setup import build_dataloader_generators, setup_torch_device, setup_workload
from model_explorer.utils.workload import Workload
from model_explorer.utils.setup import get_model_init_function, get_model_update_function
from model_explorer.result_handling.collect_results import collect_results

RESULTS_DIR = "./results"


def save_results(result_df: pd.DataFrame, problem_name: str, model_name: str, dataset_name: str):
    # store results in csv
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")

    filename = 'reeval_{}_{}_{}_{}.csv'.format(
        problem_name, model_name, dataset_name, date_str)
    filename = os.path.join(RESULTS_DIR, filename)
    result_df.to_csv(filename)
    logger.info(f"Saved result object to: {filename}")


def reevaluate_individuals(workload: Workload, results_path: str,
                           count: int, progress: bool,
                           verbose: bool):

    dataloaders = build_dataloader_generators(
        workload['reevaluation']['datasets'])
    reevaluate_dataloader = dataloaders['reevaluate']
    model, accuracy_function = setup_workload(workload['model'])
    device = setup_torch_device()

    model_init_func = get_model_init_function(workload['problem']['problem_function'])
    model_update_func = get_model_update_function(workload['problem']['problem_function'])
    kwargs: dict = workload['exploration']['extra_args']
    if 'calibration' in workload.yaml_data:
        kwargs['calibration_file'] = workload['calibration']['file']
    explorable_model = model_init_func(model, device, verbose, **kwargs)

    # Load the specified results file and pick n individuals
    results_collection = collect_results(results_path)

    results_collection.drop_duplicate_parameters()
    logger.debug("Loaded in total {} individuals".format(
        len(results_collection.individuals)))

    # select individuals with limit and cost function
    individuals = results_collection.get_better_than_individuals(0.72)

    results = pd.DataFrame(columns=[
        'generation', 'individual', 'accuracy', 'acc_full', 'F_0',
        'mutation_eta', 'mutation_prob', 'crossover_eta', 'crossover_prob',
        'selection_press'
    ])

    # sort the individuals for cost and select n with lowest cost
    individuals = individuals[:count]

    logger.info(
        "Selecting {} individual(s) for reevaluation with the full dataset.".
        format(len(individuals)))

    for i, individual in enumerate(individuals):
        logger.debug(
            "Evaluating {} / {} models with optimization accuracy: {}".format(
                i + 1, len(individuals), individual.accuracy))
        model_update_func(explorable_model, individual.parameter)
        full_accuracy = accuracy_function(explorable_model.base_model,
                                          reevaluate_dataloader,
                                          progress=progress,
                                          title="Reevaluating {}/{}".format(
                                              i + 1,
                                              len(individuals)))

        logger.info(
            "Done with ind {} / {}, accuracy is {:.4f}, was before {:.4f}, fo={}"
            .format(i + 1, len(individuals), full_accuracy,
                    individual.accuracy, individual.further_objectives))

        loc_dict = individual.to_dict_without_parameters()
        loc_dict['acc_full'] = full_accuracy.item()
        # loc_dict['bits'] = individual.bits
        # loc_dict['cost'] = cost
        results.loc[i] = loc_dict

    return results


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

    workload_file = opt.workload
    if os.path.isfile(workload_file):
        workload = Workload(workload_file)
        results = reevaluate_individuals(workload,
                                         opt.results_path, opt.top_elements,
                                         opt.progress, opt.verbose)
        save_results(
            results, workload['problem']['problem_function'], workload['model']['type'],
            workload['reevaluation']['datasets']['reevaluate']['type'])

    else:
        logger.warning("Declared workload file could not be found.")
        raise Exception(f"No file {opt.workload} found.")

    logger.info("Reevaluation of individuals finished")
