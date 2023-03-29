import os
import sys
import argparse
import subprocess

# FIXME?
sys.path.append(os.path.join(os.path.dirname(__file__), '../model_explorer/third_party/yolop_det_seg'))

import pandas as pd

from tqdm import tqdm
from torchinfo import summary
from torchinfo.layer_info import LayerInfo
from typing import List

from model_explorer.utils.logger import logger, set_console_logger_level
from model_explorer.utils.workload import Workload
from model_explorer.result_handling.collect_results import collect_results
from model_explorer.utils.setup import setup_workload, get_model_init_function, setup_torch_device

from model_explorer.third_party.timeloop.scripts.timeloop import rewrite_workload_bounds
from model_explorer.third_party.timeloop.scripts.parse_timeloop_output import parse_timeloop_stats

from pytorch_quantization.nn import QuantConv2d



def select_individuals(results_path: str, count: int) -> pd.DataFrame:
    results_collection = collect_results(results_path)

    results_collection.drop_duplicate_parameters()
    logger.debug("Loaded in total {} distinct individuals".format(
        len(results_collection.individuals)))

    # select individuals based on a prodcut of normed F_0 and accuracy
    ind_df = results_collection.to_dataframe()

    # ind_df['F_0'] = -ind_df['F_0']
    # ind_df['norm_f0'] = ind_df['F_0'] / ind_df['F_0'].max()
    # ind_df['norm_acc'] = ind_df['accuracy'] / ind_df['accuracy'].max()
    # ind_df['weighted'] = ind_df['norm_f0'] * ind_df['norm_acc']

    ind_filtered = ind_df.sort_values(by=['F_0'], ascending=True)
    ind_filtered = ind_filtered[0:count]  # .head(count)

    return ind_filtered


def run_timeloop(layer_info):
    pass


def compute_memory_saving(workload: Workload, individuals: list):
    timeloop_binary = os.path.abspath(os.path.join(os.path.dirname(__file__), '../model_explorer/third_party/timeloop/bin/timeloop-mapper'))

    if not os.path.exists(timeloop_binary):
        raise FileNotFoundError("Please build timeloop first (see model_explorer/third_party/timeloop)")

    # FIXME: input size parametrizable
    input_size = (3, 640, 640)  # C, H, W; N = 1

    model, _ = setup_workload(workload['model'])
    device = setup_torch_device()
    model_init_func = get_model_init_function(workload['problem']['problem_function'])

    # Build model with custom modules 
    kwargs: dict = workload['exploration']['extra_args']
    if 'calibration' in workload.yaml_data:
        kwargs['calibration_file'] = workload['calibration']['file']
    explorable_model = model_init_func(model, device, **kwargs)

    layer_list = summary(model, input_size=(1, 3, 640, 640))

    timeloop_layers: List[LayerInfo] = []
    for layer_info in layer_list.summary_list:
        if isinstance(layer_info.module, QuantConv2d):
            timeloop_layers.append(layer_info)

    print(f"working on {len(timeloop_layers)} layers with timeloop")

    layer: LayerInfo
    results = {}
    for i, layer in enumerate(tqdm(timeloop_layers, ascii=True)):
        input_bits = layer.module._input_quantizer._num_bits
        weights_bits = layer.module._weight_quantizer._num_bits

        workload_bounds = [
            layer.input_size[2],
            layer.input_size[3],
            layer.input_size[1],
            layer.input_size[0],
            layer.output_size[1],
            layer.kernel_size[0],
            layer.kernel_size[1],
            layer.module.padding[0],
            layer.module.padding[1],
            layer.module.stride[0],
            layer.module.stride[1]
        ]

        config_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../timeloop_eval/workload.yaml'))
        altered_config_file = os.path.join(os.path.dirname(config_file), f"layer_{i}.yaml")
        projected_results_file = os.path.join(os.path.dirname(config_file), "timeloop-mapper.map+stats.xml")
        timeloop_cwd = os.path.join(os.path.dirname(config_file))
        timeloop_log_file = os.path.join(os.path.dirname(config_file), "timeloop.log")
        rewrite_workload_bounds(config_file, altered_config_file, workload_bounds)

        with open(timeloop_log_file, "w") as outfile:
            subprocess.call([timeloop_binary, altered_config_file], cwd=timeloop_cwd, stdout=outfile, stderr=outfile)

        timeloop_result = parse_timeloop_stats(projected_results_file)

        results[i] = {'LayerIdx': i,
                      'LayerType': layer.class_name,
                      'Energy': timeloop_result['energy_breakdown_pJ']['DRAM']['energy'],
                      'Reads': timeloop_result['energy_breakdown_pJ']['DRAM']['reads_per_instance'],
                      'Updates': timeloop_result['energy_breakdown_pJ']['DRAM']['updates_per_instance'],
                      'BitsInp': input_bits,
                      'BitsW': weights_bits,
                      'Workload_Size': workload_bounds}

    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv('results.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("workload_file")
    parser.add_argument("results_path")
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

    if opt.verbose:
        set_console_logger_level(level=logging.DEBUG)

    # logger.info("Memory evaluation of {} individuals started".format(
    #     opt.top_elements))

    workload_file = opt.workload_file

    if not os.path.isfile(workload_file):
        logger.warning("Declared workload file could not be found.")
        raise Exception(f"No file {opt.workload} found.")

    workload = Workload(workload_file)
    # individuals = select_individuals(opt.results_path, opt.top_elements)

    compute_memory_saving(workload, None)
    # save_results_df_to_csv('retrain', results, workload['problem']['problem_function'],
                        #    workload['model']['type'], workload['reevaluation']['datasets']['reevaluate']['type'])

    # logger.info("Retraining Process Finished")

