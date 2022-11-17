import sys
import os
import argparse
import numpy as np
from datetime import datetime
import pandas as pd

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
from src.utils.setup import setup
from src.utils.workload import Workload
from src.exploration.weighting_functions import bits_weighted_linear
from src.utils.data_loader_generator import DataLoaderGenerator
from src.utils.pickeling import CPUUnpickler

from src.exploration.problems import LayerwiseQuantizationProblem
from src.quantization.quantized_model import QuantizedModel


def get_fitted_individuals(pkl_file) -> pd.DataFrame:
    with open(pkl_file, 'rb') as f:
        d = CPUUnpickler(f).load()
    
    accuracy_bound = d.problem.min_accuracy
    shape_x = d.problem.n_obj + d.problem.n_constr + d.problem.n_var

    all_individuals = np.empty( (0, shape_x) )

    for h in d.history:
        for ind in h.opt:
            individual_row = np.concatenate( (ind.get("G") - accuracy_bound, ind.get("F"), ind.get("X")) )
            individual_row = np.expand_dims(individual_row, axis=0)
            all_individuals = np.concatenate( (all_individuals, individual_row), axis=0)
        
    # remove all rows with nan values
    all_individuals = all_individuals[~np.isnan(all_individuals).any(axis=1)]
            
    df = pd.DataFrame(all_individuals)
    df[0] = -df[0] # invert Accuracy
    df_fit = df.where(df[0] > accuracy_bound) 
    return df_fit


def retrain_best_individuals(workload, calibration_file, results_file, progress=True, verbose=True):
    model, accuracy_function, validation_dataset, train_dataset, \
        training_items, collate_fn, device = setup(workload)

    train_dataloader_generator = DataLoaderGenerator(train_dataset, 
                                                     collate_fn, 
                                                     items=training_items,
                                                     batch_size=workload['retraining']['batch_size'],
                                                     limit=workload['retraining']['sample_limit'])
    validation_dataloader_generator = DataLoaderGenerator(validation_dataset, 
                                                          collate_fn, 
                                                          batch_size=workload['retraining']['batch_size'])
                                                        #   limit=workload['retraining']['sample_limit'])

    fit_individuals = get_fitted_individuals(results_file)
    best_individuals = fit_individuals.sort_values([0, 2],
                                                   ascending=[False, True]) #sort by accuracy and then by #bits 
    
    counter = 0
    for index, row in best_individuals.iterrows():
        counter += 1

        qmodel = QuantizedModel(model, device=device, verbose=verbose)
        qmodel.load_parameters(calibration_file)
        
        # Update Model 
        qmodel.bit_widths = row[3:].to_numpy()
        logger.info("Starting new model (ID #{}) starting acc {:.3f}, starting bits: {}".format(index, row[0], row[2]))

        qmodel.retrain(train_dataloader_generator=train_dataloader_generator,
                       test_dataloader_generator=validation_dataloader_generator,
                       accuracy_function=accuracy_function,
                       num_epochs=workload['retraining']['epochs'],
                       progress=progress)
        qmodel.save_parameters('retrained_model_{}.pkl'.format(index))

        logger.info("Saved retrained Model")
        
        # stop after the amount of individuals ...
        if counter >= workload['retraining']['n_best']:
            break



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("calibration_file")
    parser.add_argument("workload_file")
    parser.add_argument("results_file")
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
        "-fn",
        "--filename",
        help="override default filename for calibration pickle file")
    opt = parser.parse_args()

    logger.info("Quantization Exploration Started")

    workload_file = opt.workload_file
    if os.path.isfile(workload_file):
        workload = Workload(workload_file)
        retrain_best_individuals(workload, opt.calibration_file, opt.results_file, opt.progress, opt.verbose)

    else:
        logger.warning("Declared workload file could not be found.")
        raise Exception(f"No file {opt.workload} found.")

    logger.info("Quantization Exploration Finished")
