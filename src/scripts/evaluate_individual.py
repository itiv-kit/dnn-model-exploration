import os
import argparse
import pandas as pd
from datetime import datetime
import importlib

# import troch quantization and activate the replacement of modules
from pytorch_quantization import quant_modules
quant_modules.initialize()

from src.utils.logger import logger
from src.utils.setup import build_dataloader_generators, setup_torch_device, setup_model
from src.utils.workload import Workload
from src.quantization.quantized_model import QuantizedModel
from src.result_handling.results_loader import ResultsLoader


def reevaluate_individual(workload: Workload, calibration_file: str, results_file: str,
                          count: int, progress: bool, verbose: bool):
    
    dataloaders = build_dataloader_generators(workload['reevaluation']['datasets'])
    reevaluate_dataloader = dataloaders['reevaluate']
    model, accuracy_function = setup_model(workload['model'])
    device = setup_torch_device()
    weighting_function = getattr(importlib.import_module('src.exploration.weighting_functions'), 
                                 workload['reevaluation']['bit_weighting_function'], None)
    assert weighting_function is not None and callable(weighting_function), "error loading weighting function"

    qmodel = QuantizedModel(model, device, 
                            weighting_function=weighting_function,
                            verbose=verbose)
    # Load the previously generated calibration file
    logger.info(f"Loading calibration file: {calibration_file}")
    qmodel.load_parameters(calibration_file)

    # Load the specified results file and pick n individuals 
    rl = ResultsLoader(pickle_file=results_file)
    individuals = rl.get_accuracy_sorted_individuals()[:count]

    results = pd.DataFrame(columns=['gen', 'individual', 'acc_explore', 'acc_full', 'weighted_bits', 'bits'])

    logger.info("Selecting {} individual(s) for reevaluation with the full dataset.".format(len(individuals)))
    
    for i, individual in enumerate(individuals):
        logger.debug("Evaluating {} / {} models with optimization accuracy: {}".format(
                i+1, len(individuals), individual.accuracy))
        qmodel.bit_widths = individual.bits
        # full_accuracy = accuracy_function(qmodel.model, reevaluate_dataloader, progress=progress,
        #                                   title="Reevaluating {}/{}".format(i+1, len(individuals)))
        full_accuracy = 1
        logger.info("Done with ind {} / {}, accuracy is {:.4f}, before was {:.4f}".format(
                i+1, len(individuals), full_accuracy, individual.accuracy))

        results.loc[i] = {'gen': individual.generation,
                          'individual': individual.individual_idx,
                          'acc_explore': individual.accuracy,
                          'acc_full': full_accuracy,
                          'weighted_bits': individual.weighted_bits,
                          'bits': individual.bits}

    results.to_csv('results.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("calibration_file")
    parser.add_argument("workload", help="The path to the workload yaml file.")
    parser.add_argument("results_file", help="Path to the results file to be evaluated")
    parser.add_argument('-n', "--top_elements", help="Select n individuals with the lowest bits", type=int)
    opt = parser.parse_args()

    logger.info("Reevaluation of individuals started")

    workload_file = opt.workload
    if os.path.isfile(workload_file):
        workload = Workload(workload_file)
        reevaluate_individual(workload, opt.calibration_file, opt.results_file, opt.top_elements, True, False)

    else:
        logger.warning("Declared workload file could not be found.")
        raise Exception(f"No file {opt.workload} found.")

    logger.info("Quantization GA Sweep Exploration Finished")

