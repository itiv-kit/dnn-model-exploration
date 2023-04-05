import os
import sys
import argparse
import logging

from model_explorer.utils.logger import logger, set_console_logger_level
from model_explorer.utils.setup import build_dataloader_generators, setup_workload, setup_torch_device
from model_explorer.utils.workload import Workload
from model_explorer.models.quantized_model import QuantizedModel

from pytorch_quantization.tensor_quant import QuantDescriptor


def generate_calibration(workload: Workload, progress: bool, filename: str):
    dataloaders = build_dataloader_generators(workload['calibration']['datasets'])
    model, _ = setup_workload(workload['model'])
    device = setup_torch_device()

    dataset_gen = dataloaders['calibrate']

    quant_descriptor = QuantDescriptor(calib_method='histogram')
    qmodel = QuantizedModel(model, device, quantization_descriptor=quant_descriptor)

    qmodel.generate_calibration_file(dataset_gen.get_dataloader(), progress, calib_method='histogram',
                                     method='percentile', percentile=99.99)

    qmodel.save_parameters(filename)


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
        "-f",
        "--force",
        help="Force overwrite file, if exists",
        action="store_true"
    )

    opt = parser.parse_args()

    if opt.verbose:
        set_console_logger_level(level=logging.DEBUG)

    logger.info("Calibration Started")

    workload_file = opt.workload
    if not os.path.isfile(workload_file):
        logger.warning("Declared workload file could not be found.")
        raise Exception(f"No file {opt.workload} found.")

    workload = Workload(workload_file)

    if "quant" not in workload['problem']['problem_function']:
        logger.warning("The selected workload is not a quantization workload!")

    filename = workload['calibration']['file']
    if os.path.exists(filename) and opt.force is False:
        logger.warning("Calibration file already exists, stopping")
        sys.exit(0)

    generate_calibration(workload, opt.progress, filename)

    logger.info("Calibtration Finished")

