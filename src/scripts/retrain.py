import sys
import os
import argparse
import numpy as np
from datetime import datetime
import pandas as pd
import glob

# import troch quantization and activate the replacement of modules
from pytorch_quantization import quant_modules
quant_modules.initialize()

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.evaluator import Evaluator

from src.utils.logger import logger
from src.utils.setup import setup_model, setup_torch_device, build_dataloader_generators
from src.utils.workload import Workload
from src.result_handling.results_collection import ResultsCollection
from src.result_handling.collect_results import collect_results
from src.utils.data_loader_generator import DataLoaderGenerator
from src.utils.pickeling import CPUUnpickler

from src.exploration.problems import LayerwiseQuantizationProblem
from src.quantization.quantized_model import QuantizedModel



RESULTS_DIR = "./results"


def save_results(result_df:pd.DataFrame, model_name:str, dataset_name:str):
    # store results in csv
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    filename = 'retrain_results_{}_{}_{}.csv'.format(
        model_name, dataset_name, date_str)
    result_df.to_csv(filename)
    logger.info(f"Saved result object to: {filename}")


def retrain_best_individuals(workload, calibration_file, results_path, count, progress=True, verbose=True):

    datasets = build_dataloader_generators(workload['retraining']['datasets'])
    reeval_datasets = build_dataloader_generators(workload['reevaluation']['datasets'])
    reevaluate_dataloader = reeval_datasets['reevaluate']
    device = setup_torch_device()

    # Load the specified results file and pick n individuals 
    results_collection = collect_results(results_path)

    results_collection.drop_duplicate_bits()
    logger.debug("Loaded in total {} individuals".format(len(results_collection.individuals)))

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

    individuals_with_cost.sort(key=lambda ind: ind[0])
    individuals_with_cost = individuals_with_cost[:count]

    results = pd.DataFrame(columns=['generation', 'individual', 'accuracy', 'full_acc_after',
                                    'full_acc_before', 'retrain_epoch_accs',
                                    'weighted_bits', 'mutation_eta', 'mutation_prob', 
                                    'crossover_eta', 'crossover_prob', 'selection_press',
                                    'cost', 'bits'])

    logger.info("Selecting {} individual(s) for retraining.".format(len(individuals_with_cost)))
    
    for index, (cost, individual) in enumerate(individuals_with_cost):
        model, accuracy_function = setup_model(workload['model'])
        qmodel = QuantizedModel(model, device=device, verbose=verbose)
        qmodel.load_parameters(calibration_file)
        full_accuracy_before = accuracy_function(qmodel.model, reevaluate_dataloader, progress=progress,
                                                 title="Reevaluating")

        # Update Model 
        qmodel.bit_widths = individual.bits
        logger.info("Starting new model (ID #{}) starting acc {:.3f}, starting bits: {}".format(index, full_accuracy_before.float(), individual.weighted_bits))

        epoch_accs = qmodel.retrain(train_dataloader_generator=datasets['train'],
                                    test_dataloader_generator=datasets['validation'],
                                    accuracy_function=accuracy_function,
                                    num_epochs=workload['retraining']['epochs'],
                                    progress=progress)
        qmodel.save_parameters('retrained_model_{}.pkl'.format(index))
        full_accuracy_after = accuracy_function(qmodel.model, reevaluate_dataloader, progress=progress,
                                                title="Reevaluating")

        loc_dict = individual.to_dict_without_bits()
        loc_dict['full_acc_after'] = full_accuracy_after.float()
        loc_dict['full_acc_before'] = full_accuracy_before.float()
        loc_dict['retrain_epoch_acs'] = epoch_accs
        loc_dict['bits'] = individual.bits
        loc_dict['cost'] = cost
        results.loc[index] = loc_dict

        logger.info("Saved retrained Model")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("calibration_file")
    parser.add_argument("workload_file")
    parser.add_argument("results_path")
    parser.add_argument('-n', "--top_elements", help="Select n individuals with the lowest bits", type=int)
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
    opt = parser.parse_args()

    logger.info("Retraining of {} individuals started".format(opt.top_elements))

    workload_file = opt.workload_file
    if os.path.isfile(workload_file):
        workload = Workload(workload_file)
        results = retrain_best_individuals(workload, opt.calibration_file, opt.results_path, opt.top_elements, opt.progress, opt.verbose)
        save_results(results, workload['model']['type'], workload['reevaluation']['datasets']['reevaluate']['type'])

    else:
        logger.warning("Declared workload file could not be found.")
        raise Exception(f"No file {opt.workload} found.")

    logger.info("Retraining Process Finished")
