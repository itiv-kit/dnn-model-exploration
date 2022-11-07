import os
import sys
import argparse
from src.utils.logger import logger

from src.utils.setup import setup
from src.utils.workload import Workload
from src.utils.predicates import conv2d_predicate
from src.utils.data_loader_generator import DataLoaderGenerator
from src.quantization.quantized_model import QuantizedModel


def generate_calibration(workload: Workload, verbose: bool, progress: bool, filename:str):
    
    model, accuracy_function, dataset, collate_fn, device = setup(workload)
    dataset_gen = DataLoaderGenerator(dataset, collate_fn, batch_size=256, limit=10000, fixed_random=True)
    
    qmodel = QuantizedModel(model, device, conv2d_predicate, verbose=verbose)

    qmodel.run_calibration(dataset_gen.get_dataloader(), progress, calib_method='histogram', 
                           mehtod='percentile', percentile=99.99)
    
    qmodel.save_calibration(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("workload", help="The path to the workload yaml file.")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show verbose information.",
    )
    parser.add_argument(
        "-p",
        "--progress",
        action="store_true",
        help="Show the current inference progress.",
    )
    parser.add_argument(
        "-fn",
        "--filename",
        help="override default filename for calibration pickle file"
    )
    parser.add_argument(
        "-f",
        "--force",
        help="Force overwrite file, if exists",
        action="store_true"
    )

    opt = parser.parse_args()
    
    logger.info("Calibration Started")

    workload_file = opt.workload
    if os.path.isfile(workload_file):
        workload = Workload(workload_file)

        filename = 'calib_{}_{}.pkl'.format(workload['model']['type'], workload['dataset']['type'])
        if opt.filename:
            filename = opt.filename
        if os.path.exists(filename) and opt.force is False:
            logger.warning("Calibration file already exists, stopping")
            sys.exit(0)
        
        generate_calibration(workload, opt.verbose, opt.progress, filename)
    else:
        logger.warning("Declared workload file could not be found.")
        raise Exception(f"No file {opt.workload} found.")
    
    logger.info("Calibtration Finished")

