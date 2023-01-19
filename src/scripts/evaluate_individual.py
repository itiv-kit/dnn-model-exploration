import os
import argparse
import pandas as pd
import importlib
from datetime import datetime

# import troch quantization and activate the replacement of modules
from pytorch_quantization import quant_modules

quant_modules.initialize()

from src.utils.logger import logger
from src.utils.setup import build_dataloader_generators, setup_torch_device, setup_workload
from src.utils.workload import Workload
from src.quantization.quantized_model import QuantizedModel
from src.result_handling.collect_results import collect_results

RESULTS_DIR = "./results"


def save_results(result_df: pd.DataFrame, model_name: str, dataset_name: str):
    # store results in csv
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    filename = 'reevaluation_results_{}_{}_{}.csv'.format(
        model_name, dataset_name, date_str)
    result_df.to_csv(filename)
    logger.info(f"Saved result object to: {filename}")


def reevaluate_individuals(workload: Workload, calibration_file: str,
                           results_path: str, count: int, progress: bool,
                           verbose: bool):

    dataloaders = build_dataloader_generators(
        workload['reevaluation']['datasets'])
    reevaluate_dataloader = dataloaders['reevaluate']
    model, accuracy_function = setup_workload(workload['model'])
    device = setup_torch_device()
    weighting_function = getattr(
        importlib.import_module('src.exploration.weighting_functions'),
        workload['reevaluation']['bit_weighting_function'], None)
    assert weighting_function is not None and callable(
        weighting_function), "error loading weighting function"

    qmodel = QuantizedModel(model,
                            device,
                            weighting_function=weighting_function,
                            verbose=verbose)
    # Load the previously generated calibration file
    logger.info(f"Loading calibration file: {calibration_file}")
    qmodel.load_parameters(calibration_file)

    # Load the specified results file and pick n individuals
    results_collection = collect_results(results_path)

    results_collection.drop_duplicate_bits()
    logger.debug("Loaded in total {} individuals".format(
        len(results_collection.individuals)))

    # select individuals with limit and cost function
    individuals = results_collection.get_better_than_individuals(0.72)
    # get max and min for normalize
    max_weighted_bits = max([ind.weighted_bits for ind in individuals])
    min_weighted_bits = min([ind.weighted_bits for ind in individuals])
    max_accuracy = max([ind.accuracy for ind in individuals])
    min_accuracy = min([ind.accuracy for ind in individuals])

    individuals_with_cost = []
    for ind in individuals:
        cost = 0.5*((ind.weighted_bits - min_weighted_bits) / (max_weighted_bits - min_weighted_bits)) + \
               0.5-(0.5*((ind.accuracy - min_accuracy) / (max_accuracy - min_accuracy)))
        individuals_with_cost.append((cost, ind))

    results = pd.DataFrame(columns=[
        'generation', 'individual', 'accuracy', 'acc_full', 'weighted_bits',
        'mutation_eta', 'mutation_prob', 'crossover_eta', 'crossover_prob',
        'selection_press', 'cost', 'bits'
    ])

    # sort the individuals for cost and select n with lowest cost
    individuals_with_cost.sort(key=lambda ind: ind[0])
    individuals_with_cost = individuals_with_cost[:count]

    logger.info(
        "Selecting {} individual(s) for reevaluation with the full dataset.".
        format(len(individuals_with_cost)))

    for i, (cost, individual) in enumerate(individuals_with_cost):
        logger.debug(
            "Evaluating {} / {} models with optimization accuracy: {}".format(
                i + 1, len(individuals_with_cost), individual.accuracy))
        qmodel.bit_widths = individual.bits
        full_accuracy = accuracy_function(qmodel.model,
                                          reevaluate_dataloader,
                                          progress=progress,
                                          title="Reevaluating {}/{}".format(
                                              i + 1,
                                              len(individuals_with_cost)))

        logger.info(
            "Done with ind {} / {}, accuracy is {:.4f}, was before {:.4f} at cost {:.4f} ({} bit)"
            .format(i + 1, len(individuals_with_cost), full_accuracy,
                    individual.accuracy, cost, individual.weighted_bits))

        loc_dict = individual.to_dict_without_bits()
        loc_dict['acc_full'] = full_accuracy.float()
        loc_dict['bits'] = individual.bits
        loc_dict['cost'] = cost
        results.loc[i] = loc_dict

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("calibration_file")
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
        results = reevaluate_individuals(workload, opt.calibration_file,
                                         opt.results_path, opt.top_elements,
                                         opt.progress, opt.verbose)
        save_results(
            results, workload['model']['type'],
            workload['reevaluation']['datasets']['reevaluate']['type'])

    else:
        logger.warning("Declared workload file could not be found.")
        raise Exception(f"No file {opt.workload} found.")

    logger.info("Quantization GA Sweep Exploration Finished")
